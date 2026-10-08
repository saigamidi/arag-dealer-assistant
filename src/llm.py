"""
One place for chat-model calls, so the provider can be swapped (Azure today,
others later) without touching the generator or the evaluation.

    ChatClient.complete_json(system, user) -> ChatResult(data: dict, tokens_in, tokens_out)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ChatResult:
    data: dict
    tokens_in: int = 0
    tokens_out: int = 0
    raw: str = ""


class ChatClient(Protocol):
    name: str
    def complete_json(self, system: str, user: str, max_tokens: int = 500) -> ChatResult: ...


class AzureChatClient:
    """Calls an Azure OpenAI chat deployment in JSON mode."""

    def __init__(self, deployment: str | None = None, client=None):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
        if client is None:
            from openai import AzureOpenAI
            endpoint, key = os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")
            if not (endpoint and key and self.deployment):
                raise RuntimeError("Azure OpenAI secrets are missing (see scripts/check_azure.py).")
            client = AzureOpenAI(azure_endpoint=endpoint, api_key=key,
                                 api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                                 timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")), max_retries=2)
        self.client = client
        self.name = f"azure:{self.deployment}"
        self._reasoning_params = False   # set if the model rejects max_tokens/temperature
        self._use_seed = True            # dropped if the deployment rejects it

    def complete_json(self, system: str, user: str, max_tokens: int = 500) -> ChatResult:
        try:
            return self._complete(system, user, max_tokens)
        except Exception as exc:
            if self._use_seed and "seed" in str(exc):
                self._use_seed = False
                return self._complete(system, user, max_tokens)
            raise

    def _complete(self, system: str, user: str, max_tokens: int) -> ChatResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs = {"model": self.deployment, "messages": messages,
                  "response_format": {"type": "json_object"}}
        if self._use_seed:
            kwargs["seed"] = 42           # best-effort repeatability; not a guarantee
        if self._reasoning_params:
            resp = self.client.chat.completions.create(**kwargs, max_completion_tokens=max_tokens * 4)
        else:
            try:
                resp = self.client.chat.completions.create(**kwargs, max_tokens=max_tokens, temperature=0)
            except Exception as exc:
                # Newer reasoning models use max_completion_tokens and a fixed temperature.
                msg = str(exc)
                if "max_tokens" in msg or "temperature" in msg:
                    self._reasoning_params = True
                    resp = self.client.chat.completions.create(**kwargs, max_completion_tokens=max_tokens * 4)
                else:
                    raise
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        usage = getattr(resp, "usage", None)
        return ChatResult(data=data, raw=raw,
                          tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                          tokens_out=getattr(usage, "completion_tokens", 0) or 0)


@dataclass
class FakeChatClient:
    """Test double: returns scripted JSON responses and records the prompts it saw."""
    responses: list[dict]
    name: str = "fake"
    prompts: list[tuple[str, str]] = field(default_factory=list)
    fail: bool = False

    def complete_json(self, system: str, user: str, max_tokens: int = 500) -> ChatResult:
        self.prompts.append((system, user))
        if self.fail:
            raise ConnectionError("LLM unavailable")
        data = self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]
        return ChatResult(data=data, tokens_in=100, tokens_out=20, raw=json.dumps(data))
