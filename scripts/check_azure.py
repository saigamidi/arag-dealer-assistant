"""
Check that this codespace can reach your Azure OpenAI (Microsoft Foundry) models.

Needs Codespaces secrets:
    AZURE_OPENAI_ENDPOINT            e.g. https://arag-openai.openai.azure.com/
    AZURE_OPENAI_API_KEY             Key 1 from "Keys and Endpoint"
    AZURE_OPENAI_CHAT_DEPLOYMENT     your chat deployment name, e.g. arag-chat
    AZURE_OPENAI_EMBED_DEPLOYMENT    your embedding deployment name, e.g. arag-embed
Optional:
    AZURE_OPENAI_API_VERSION         defaults to 2024-10-21

Run from the project root:   python scripts/check_azure.py
Costs a fraction of a cent (two tiny model calls).
"""

import os
import sys

from openai import APIConnectionError, APIStatusError, AzureOpenAI

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
CHAT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
EMBED = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

HINTS = {
    401: "The key is wrong or from a different resource. Copy Key 1 again from Keys and Endpoint.",
    403: "Access denied. Check the key belongs to this resource and the subscription is Pay-As-You-Go.",
    404: "Deployment not found. The secret must be the DEPLOYMENT name you chose, not the model name, "
         "and the endpoint must be this resource's endpoint.",
    429: "Rate limit or zero quota. Free Trial subscriptions have 0 quota for Azure OpenAI: "
         "upgrade to Pay-As-You-Go, then redeploy or raise the deployment's tokens-per-minute.",
}


def step(name, fn):
    try:
        print(f"✅ {name}: {fn()}")
        return True
    except APIStatusError as e:
        print(f"❌ {name}: HTTP {e.status_code}: {str(e.message)[:300]}")
        if e.status_code in HINTS:
            print(f"   {HINTS[e.status_code]}")
    except APIConnectionError:
        print(f"❌ {name}: could not reach {ENDPOINT}. Check the endpoint URL secret.")
    except Exception as e:  # anything unexpected
        print(f"❌ {name}: {type(e).__name__}: {e}")
    return False


def config():
    missing = [n for n, v in [("AZURE_OPENAI_ENDPOINT", ENDPOINT), ("AZURE_OPENAI_API_KEY", API_KEY),
                              ("AZURE_OPENAI_CHAT_DEPLOYMENT", CHAT), ("AZURE_OPENAI_EMBED_DEPLOYMENT", EMBED)]
               if not v]
    if missing:
        raise RuntimeError(f"missing secrets: {', '.join(missing)}. Add them, then restart the codespace.")
    return f"endpoint {ENDPOINT}, API version {API_VERSION}"


client = None


def embed():
    out = client.embeddings.create(model=EMBED, input="What is the restocking fee for vehicle returns?")
    return f"deployment '{EMBED}' returned a vector of {len(out.data[0].embedding)} numbers"


def chat():
    messages = [{"role": "user", "content": "Reply with exactly: Azure is connected."}]
    try:
        resp = client.chat.completions.create(model=CHAT, messages=messages, max_tokens=20, temperature=0)
    except APIStatusError as e:
        # Newer reasoning models use max_completion_tokens and a fixed temperature.
        if e.status_code == 400 and ("max_tokens" in str(e.message) or "temperature" in str(e.message)):
            resp = client.chat.completions.create(model=CHAT, messages=messages, max_completion_tokens=200)
        else:
            raise
    text = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    return f"deployment '{CHAT}' said \"{text}\" ({u.prompt_tokens} in / {u.completion_tokens} out tokens)"


if __name__ == "__main__":
    print()
    ok = step("Configuration", config)
    if ok:
        client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=API_KEY, api_version=API_VERSION)
        ok = step("Embeddings", embed) and step("Chat model", chat)
    print("\nAll set for the cloud phase.\n" if ok else "\nFix the first ❌ above, then run again.\n")
    sys.exit(0 if ok else 1)
