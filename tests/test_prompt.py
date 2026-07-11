import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.prompt import (
    PromptType,
    get_prompt,
)

print("=" * 60)
print("PROMPT MANAGEMENT SYSTEM TEST")
print("=" * 60)

# Load the QA prompt
qa_prompt = get_prompt(PromptType.QA)

# Format the prompt with sample data
formatted_prompt = qa_prompt.invoke(
    {
        "context": "Machine Learning is a subset of Artificial Intelligence.",
        "question": "What is Machine Learning?",
    }
)

print(f"Prompt Type : {PromptType.QA.value}")
print(f"Message Count : {len(formatted_prompt.messages)}")

print("\n" + "=" * 60)
print("FORMATTED PROMPT")
print("=" * 60)

for i, message in enumerate(formatted_prompt.messages, start=1):
    print(f"\n{'=' * 50}")
    print(f"MESSAGE {i} ({message.type.upper()})")
    print("=" * 50)
    print(message.content)

print("\n" + "=" * 60)
print("PROMPT TEST COMPLETED SUCCESSFULLY")
print("=" * 60)