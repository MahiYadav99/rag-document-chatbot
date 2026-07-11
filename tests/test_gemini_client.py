import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

from llm.gemini_client import GeminiClient

print("=" * 60)
print("GEMINI CLIENT TEST")
print("=" * 60)

client = GeminiClient(
    api_key=GEMINI_API_KEY,
    model_name=GEMINI_MODEL,
)

print("Testing Gemini connection...\n")

if client.test_connection():

    print("Connection Successful!\n")

    response = client.generate_response(
        "What is Artificial Intelligence? Answer in one sentence."
    )

    print("=" * 60)
    print("MODEL RESPONSE")
    print("=" * 60)

    print(response.text)

else:

    print("Connection Failed.")