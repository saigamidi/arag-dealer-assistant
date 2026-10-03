"""
Clarification follow-ups: connect a short reply to the question it answers.

    Agent:     Can you check stock for the front brake caliper?
    Assistant: Which part do you mean: BRK-1020 (front, left) or BRK-1021 (front, right)?
    Agent:     BRK-1020                <- on its own, this has no intent
    Resolved:  "Can you check stock for the front brake caliper? (BRK-1020)"

Scope: this resolves ONE pending clarification. General multi-turn memory
("and what about its price?") stays out of scope for v1, as in the PRD.

The assistant itself stays stateless: the UI keeps the PendingClarification
from the previous answer and passes it back with the next message. Stateless
services are simpler to scale and deploy in the cloud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.router import ORDER_ID, PART_ID, POLICY_PATTERNS, REALTIME_PATTERNS, _matches

MAX_FOLLOW_UP_WORDS = 6        # longer messages are treated as new questions
ORDINALS = {"1": 0, "first": 0, "1st": 0, "one": 0,
            "2": 1, "second": 1, "2nd": 1, "two": 1,
            "3": 2, "third": 2, "3rd": 2, "three": 2}
# "right" is deliberately excluded: in this domain it usually means right-hand side.
YES = {"yes", "yeah", "yep", "correct", "y", "ok", "okay", "sure", "that one"}
FILLER = {"the", "a", "one", "please", "it", "is", "that", "this", "option", "number", "no", "id"}


@dataclass
class PendingClarification:
    original_query: str
    missing: str                              # "part" or "order"
    answer_id: str
    candidates: list[dict] = field(default_factory=list)
    suggested_id: str | None = None


def _ids_in(text: str) -> list[str]:
    parts = [f"{a.upper()}-{b}" for a, b in PART_ID.findall(text) if a.upper() != "ORD"]
    orders = [f"ORD-{n}" for n in ORDER_ID.findall(text)]
    return list(dict.fromkeys(parts + orders))


def _pick_candidate(message: str, pending: PendingClarification) -> str | None:
    words = [w for w in re.findall(r"[a-z0-9]+", message.lower())]
    meaningful = [w for w in words if w not in FILLER]

    # "yes" accepts the suggestion, or the only candidate.
    if " ".join(words) in YES or (meaningful and all(w in YES for w in meaningful)):
        if pending.suggested_id:
            return pending.suggested_id
        if len(pending.candidates) == 1:
            return pending.candidates[0]["part_id"]
        return None

    # "the first one", "2", "second"
    for w in words:
        if w in ORDINALS and ORDINALS[w] < len(pending.candidates):
            return pending.candidates[ORDINALS[w]]["part_id"]

    # "left", "the right one": match words against candidate descriptions.
    if meaningful:
        matches = [c for c in pending.candidates
                   if all(w in c["description"].lower() for w in meaningful)]
        if len(matches) == 1:
            return matches[0]["part_id"]
    return None


def resolve_follow_up(message: str, pending: PendingClarification | None) -> str | None:
    """
    If `message` answers the pending clarification, return the combined question
    to route. Return None if it's a new, standalone question.
    """
    if pending is None:
        return None
    text = message.strip()
    if len(text.split()) > MAX_FOLLOW_UP_WORDS:
        return None
    # A reply with its own intent ("price of BRK-1020?") is a new question.
    if _matches(REALTIME_PATTERNS, text) or _matches(POLICY_PATTERNS, text):
        return None

    ids = _ids_in(text)
    if not ids and pending.missing == "order":
        bare = re.fullmatch(r"#?\s*(\d{5})", text)
        if bare:
            ids = [f"ORD-{bare.group(1)}"]
    if not ids:
        chosen = _pick_candidate(text, pending)
        ids = [chosen] if chosen else []

    if not ids:
        return None
    return f"{pending.original_query} ({', '.join(ids)})"
