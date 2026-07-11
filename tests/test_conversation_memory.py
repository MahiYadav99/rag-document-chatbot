import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from memory.conversation_memory import ConversationMemory

print("=" * 60)
print("CONVERSATION MEMORY TEST")
print("=" * 60)

memory = ConversationMemory(max_messages=4)

memory.add_user_message(
    "What is Machine Learning?"
)

memory.add_assistant_message(
    "Machine Learning is a branch of Artificial Intelligence."
)

memory.add_user_message(
    "Who proposed it?"
)

memory.add_user_message("Q1")
memory.add_assistant_message("A1")

memory.add_user_message("Q2")
memory.add_assistant_message("A2")

memory.add_user_message("Q3")
memory.add_assistant_message("A3")

history = memory.get_history()

print(f"Messages Stored : {len(history)}")

print("\nConversation History\n")

for message in history:

    print(f"{message.role.upper()}: {message.content}")

print("\n" + "=" * 60)
print("FORMATTED CONVERSATION HISTORY")
print("=" * 60)

print(memory.format_history())

memory.clear()

print("\nMemory Cleared")

print(
    f"Messages Remaining : {len(memory.get_history())}"
)