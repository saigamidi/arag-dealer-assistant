"""
Shared test setup: tests never call cloud services. Even in a codespace that
has Azure secrets, every test runs with the free keyword retriever and the
stub generator unless a test explicitly builds something else.
"""

import pytest


@pytest.fixture(autouse=True)
def offline_defaults(monkeypatch):
    monkeypatch.setenv("RETRIEVER", "bm25")
    monkeypatch.setenv("GENERATOR", "stub")
