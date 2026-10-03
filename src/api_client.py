"""
Client for the live systems (the mock API in v1).

Every call returns an ApiResult instead of raising, so the assistant can
always respond, even when the source system is down or slow
(PRD > Risks: "Real-time API downtime or slow responses").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests

DEFAULT_BASE_URL = os.getenv("MOCK_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "5"))   # PRD latency target: < 5s


@dataclass
class ApiResult:
    endpoint: str                 # "inventory" or "orders"
    id: str
    ok: bool
    data: dict = field(default_factory=dict)
    error_code: str | None = None          # e.g. PART_NOT_FOUND, SERVICE_UNAVAILABLE, TIMEOUT
    message: str | None = None
    status_code: int | None = None


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT, http=None):
        # `http` lets tests pass FastAPI's TestClient instead of real HTTP.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.http = http or requests

    def _get(self, path: str, params: dict) -> tuple[int | None, dict | None, str | None]:
        try:
            r = self.http.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
            return r.status_code, r.json(), None
        except requests.Timeout:
            return None, None, "TIMEOUT"
        except requests.ConnectionError:
            return None, None, "CONNECTION_ERROR"
        except Exception as exc:  # any other transport problem (e.g. invalid JSON)
            name = type(exc).__name__.lower()
            return None, None, "TIMEOUT" if "timeout" in name else "CONNECTION_ERROR"

    def call(self, endpoint: str, id: str, warehouse: str | None = None,
             simulate: str | None = None) -> ApiResult:
        params = {k: v for k, v in {"warehouse": warehouse, "simulate": simulate}.items() if v}
        path = f"/inventory/{id}" if endpoint == "inventory" else f"/orders/{id}"
        status, body, transport_error = self._get(path, params)

        if transport_error:
            system = "inventory" if endpoint == "inventory" else "order"
            reason = "did not respond in time" if transport_error == "TIMEOUT" else "could not be reached"
            return ApiResult(endpoint, id, False, error_code=transport_error,
                             message=f"The {system} system {reason}. Please check it directly or try again shortly.")
        if status == 200:
            return ApiResult(endpoint, id, True, data=body, status_code=200)
        err = (body or {}).get("error", {})
        return ApiResult(endpoint, id, False, error_code=err.get("code", "UNKNOWN_ERROR"),
                         message=err.get("message", "Unexpected error from the source system."),
                         status_code=status)

    def health(self) -> bool:
        try:
            return self.http.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except Exception:
            return False
