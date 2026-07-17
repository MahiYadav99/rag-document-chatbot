import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from memory.conversation_memory import ConversationMemory

print("=" * 70)
print("CONVERSATION MEMORY TEST")
print("=" * 70)

# ------------------------------------------------------------------
# Create Memory
# ------------------------------------------------------------------

memory = ConversationMemory(max_messages=4)

print(f"\nMaximum Messages Allowed : {memory.max_messages}")

# ------------------------------------------------------------------
# Initial Conversation
# ------------------------------------------------------------------

print("\nAdding conversation...\n")

memory.add_user_message(
    "What is Machine Learning?"
)

memory.add_assistant_message(
    "Machine Learning is a branch of Artificial Intelligence."
)

memory.add_user_message(
    "Who proposed it?"
)

memory.add_assistant_message(
    "Arthur Samuel proposed one of the earliest definitions of Machine Learning in 1959."
)

# ------------------------------------------------------------------
# Display Stored Messages
# ------------------------------------------------------------------

history = memory.get_history()

print("=" * 70)
print("CURRENT MEMORY")
print("=" * 70)

print(f"Messages Stored : {len(history)}\n")

for i, message in enumerate(history, start=1):

    print(f"Message {i}")
    print("-" * 70)
    print(f"Role    : {message.role.capitalize()}")
    print(f"Content : {message.content}\n")

# ------------------------------------------------------------------
# Demonstrate Memory Trimming
# ------------------------------------------------------------------

print("=" * 70)
print("ADDING MORE MESSAGES (Memory Trim Demo)")
print("=" * 70)

memory.add_user_message(
    "What is Deep Learning?"
)

memory.add_assistant_message(
    "Deep Learning is a subset of Machine Learning."
)

history = memory.get_history()

print(f"\nMessages Stored After Trim : {len(history)}")
print(f"Maximum Allowed            : {memory.max_messages}\n")

for i, message in enumerate(history, start=1):

    print(f"Message {i}")
    print("-" * 70)
    print(f"Role    : {message.role.capitalize()}")
    print(f"Content : {message.content}\n")

# ------------------------------------------------------------------
# Formatted History
# ------------------------------------------------------------------

print("=" * 70)
print("FORMATTED HISTORY")
print("=" * 70)

print(memory.format_history())

# ------------------------------------------------------------------
# Clear Memory
# ------------------------------------------------------------------

memory.clear()

print("\n" + "=" * 70)
print("CLEAR MEMORY")
print("=" * 70)

print("Conversation memory cleared.")

print(f"\nMessages Remaining : {len(memory.get_history())}")