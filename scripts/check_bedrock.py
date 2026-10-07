"""
Check that this codespace can reach Amazon Bedrock with your restricted key.

Needs Codespaces secrets: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
Run from the project root:   python scripts/check_bedrock.py

Costs a fraction of a cent (two tiny model calls).
"""

import json
import os
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
CHAT_MODEL = os.getenv("BEDROCK_CHAT_MODEL", "amazon.nova-lite-v1:0")


def step(name, fn):
    try:
        result = fn()
        print(f"✅ {name}: {result}")
        return True
    except NoCredentialsError:
        print(f"❌ {name}: no AWS credentials found. Add the Codespaces secrets, then rebuild or restart the codespace.")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        print(f"❌ {name}: {code}: {e.response['Error']['Message']}")
        if code == "AccessDeniedException":
            print("   Check the IAM policy on your arag-app user, and that the model is available in this region.")
        if code in ("ValidationException", "ResourceNotFoundException"):
            print("   The model ID may differ in your region. Check the Bedrock model catalog and set it as an env var.")
    return False


def whoami():
    arn = boto3.client("sts", region_name=REGION).get_caller_identity()["Arn"]
    if arn.endswith(":root"):
        raise SystemExit("❌ You are using ROOT credentials. Create the restricted arag-app user instead.")
    return arn


def embed():
    client = boto3.client("bedrock-runtime", region_name=REGION)
    body = json.dumps({"inputText": "What is the restocking fee for vehicle returns?"})
    out = json.loads(client.invoke_model(modelId=EMBED_MODEL, body=body)["body"].read())
    return f"{EMBED_MODEL} returned a vector of {len(out['embedding'])} numbers"


def chat():
    client = boto3.client("bedrock-runtime", region_name=REGION)
    resp = client.converse(
        modelId=CHAT_MODEL,
        messages=[{"role": "user", "content": [{"text": "Reply with exactly: Bedrock is connected."}]}],
        inferenceConfig={"maxTokens": 20, "temperature": 0},
    )
    text = resp["output"]["message"]["content"][0]["text"].strip()
    usage = resp["usage"]
    return f'{CHAT_MODEL} said "{text}" ({usage["inputTokens"]} in / {usage["outputTokens"]} out tokens)'


if __name__ == "__main__":
    print(f"\nRegion: {REGION}\n")
    ok = step("Credentials", whoami) and step("Embeddings", embed) and step("Chat model", chat)
    print("\nAll set for the cloud phase.\n" if ok else "\nFix the first ❌ above, then run again.\n")
    sys.exit(0 if ok else 1)