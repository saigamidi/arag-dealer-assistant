"""
ARAG package.

Configuration is read from environment variables. On import, values from a
local `.env` file (project root) are loaded too, without overriding anything
already set, so Codespaces secrets always win over the file.

Keep secrets (API keys) in Codespaces secrets; use .env for non-secret
settings such as RETRIEVER. See .env.example.
"""

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:  # python-dotenv not installed: env vars still work
    pass
