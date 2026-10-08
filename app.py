"""
ARAG - Dealer Support Assistant (Streamlit UI)

Run from the project root, with the mock API running in another terminal:
    Terminal 1:  uvicorn mock_api.main:app --port 8000
    Terminal 2:  streamlit run app.py
"""

import streamlit as st

from src.api_client import ApiClient
from src.pipeline import Assistant

st.set_page_config(page_title="ARAG Dealer Support Assistant", page_icon="🔧", layout="centered")

ROUTE_BADGE = {
    "static": "📚 Knowledge base",
    "realtime": "⚡ Live data",
    "hybrid": "📚⚡ Knowledge base + live data",
    "clarify": "❓ Needs clarification",
}
SAMPLE_QUESTIONS = [
    "How long is the standard warranty on a new vehicle?",
    "How many BRK-1020 calipers are in stock?",
    "What is the status of order ORD-10388?",
    "Is ELC-3030 in stock, and can the dealer return it once opened?",
    "Can a dealer return an unopened set of floor mats?",
    "Can you check stock for the front brake caliper?",
]


@st.cache_resource
def get_assistant() -> Assistant:
    return Assistant()


def init_state():
    st.session_state.setdefault("messages", [])     # list of {"role", "content" | "answer"}
    st.session_state.setdefault("feedback", {})     # answer_id -> "up" / "down"
    st.session_state.setdefault("pending", None)    # question from a sample or candidate button
    st.session_state.setdefault("clarification", None)  # PendingClarification awaiting a reply


def render_answer(answer, idx: int, is_latest: bool):
    status_badge = {"refused": "🤷 Not in sources", "conflict": "⚠️ Conflicting sources",
                    "unverified": "❗ Unverified", "fallback": "🔁 AI unavailable, passages shown"}
    bits = [ROUTE_BADGE.get(answer.route, answer.route), f"{answer.latency_ms} ms"]
    if answer.status in status_badge:
        bits.append(status_badge[answer.status])
    if answer.tokens_in:
        bits.append(f"{answer.tokens_in + answer.tokens_out} tokens")
    st.caption("  ·  ".join(bits))
    if answer.interpreted_as:
        st.caption(f"↪️ Interpreted as: *{answer.interpreted_as}*")
    st.markdown(answer.text)

    # Quick replies for a clarifying question (only on the latest message).
    if is_latest and answer.pending and answer.pending.candidates:
        cols = st.columns(len(answer.pending.candidates))
        for col, c in zip(cols, answer.pending.candidates):
            if col.button(c["part_id"], key=f"cand_{idx}_{c['part_id']}", help=c["description"]):
                st.session_state.pending = c["part_id"]
                st.rerun()

    if answer.citations:
        with st.expander(f"Sources ({len(answer.citations)})"):
            for c in answer.citations:
                st.markdown(f"**[{c.number}] {c.title} › {c.section}**  \n"
                            f"`{c.source_file}` · relevance score {c.score}")
                st.text(c.text.split("\n\n", 1)[-1])
    if answer.api_results:
        with st.expander("Live data (raw API response)"):
            for r in answer.api_results:
                st.json(r.data if r.ok else {"error": r.error_code, "message": r.message})
    with st.expander("Why this route?"):
        for reason in answer.reasons:
            st.markdown(f"- {reason}")

    # Feedback (PRD user story US-7)
    given = st.session_state.feedback.get(answer.answer_id)
    if given:
        st.caption("Thanks for the feedback 👍" if given == "up" else "Thanks, flagged for review 👎")
    else:
        col1, col2, _ = st.columns([1, 1, 8])
        if col1.button("👍", key=f"up_{idx}", help="Helpful"):
            get_assistant().record_feedback(answer.answer_id, "up")
            st.session_state.feedback[answer.answer_id] = "up"
            st.rerun()
        if col2.button("👎", key=f"down_{idx}", help="Wrong or unhelpful"):
            get_assistant().record_feedback(answer.answer_id, "down")
            st.session_state.feedback[answer.answer_id] = "down"
            st.rerun()


def sidebar() -> str | None:
    with st.sidebar:
        st.header("🔧 ARAG")
        gen = get_assistant().generator
        st.caption("Dealer support assistant · " +
                   (f"AI answers ({gen.client.name})" if getattr(gen, "name", "") == "llm"
                    else "preview mode (passages only)"))
        retriever_label = {"ResilientVectorRetriever": "vector (semantic)", "HybridRetriever": "hybrid",
                           "BM25Retriever": "BM25 (keyword)"}
        kind = type(get_assistant().retriever).__name__
        st.markdown(f"**Search:** {retriever_label.get(kind, kind)}")

        healthy = ApiClient().health()
        st.markdown(f"**Mock API:** {'🟢 connected' if healthy else '🔴 not running'}")
        if not healthy:
            st.caption("Start it in another terminal:  \n`uvicorn mock_api.main:app --port 8000`")

        st.subheader("Try a question")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, width="stretch"):
                st.session_state.pending = q

        st.subheader("Test failure handling")
        mode = st.radio("Live API behaviour", ["Normal", "Outage (503)", "Slow (timeout)"],
                        label_visibility="collapsed")

        if st.button("Clear chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.clarification = None
            st.rerun()
    return {"Normal": None, "Outage (503)": "error", "Slow (timeout)": "slow"}[mode]


def main():
    init_state()
    simulate = sidebar()

    st.title("Dealer Support Assistant")
    st.caption("Ask about warranty, returns, parts, policies, stock levels or order status.")

    last = len(st.session_state.messages) - 1
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_answer(msg["answer"], i, is_latest=(i == last))

    question = st.chat_input("Ask a question…") or st.session_state.pending
    if question:
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Looking that up…"):
            answer = get_assistant().ask(question, simulate=simulate,
                                         pending=st.session_state.clarification)
        # Keep the clarifying question (if any) so the next reply can answer it.
        st.session_state.clarification = answer.pending
        st.session_state.messages.append({"role": "assistant", "answer": answer})
        st.rerun()


main()
