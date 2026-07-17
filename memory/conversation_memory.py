from dataclasses import dataclass


@dataclass
class ChatMessage:
    """
    Represents a single conversation message.
    """

    role: str
    content: str


class ConversationMemory:
    """
    Stores conversation history.
    """

    def __init__(
    self,
    max_messages: int = 10,
    ):
     """
     Initialize conversation memory.

     Args:
        max_messages (int):
            Maximum number of messages to retain.
     """

     self.max_messages = max_messages
     self.messages: list[ChatMessage] = []

    def _trim_memory(self) -> None:
     """
     Remove the oldest messages if the memory exceeds the limit.
     """

     while len(self.messages) > self.max_messages:
         self.messages.pop(0)


    def add_user_message(
        self,
        content: str,
    ) -> None:

        self.messages.append(
             ChatMessage(
                role="user",
                content=content,
            )
    )

        self._trim_memory()

    def add_assistant_message(
        self,
        content: str,
    ) -> None:

        self.messages.append(
            ChatMessage(
                role="assistant",
                content=content,
            )
    )

        self._trim_memory()

    def get_history(
        self,
    ) -> list[ChatMessage]:

        return self.messages.copy()
    def clear(
        self,
    ) -> None:

        self.messages.clear()

    def format_history(self) -> str:
         """
         Convert the conversation history into a formatted string
         that can be inserted into the prompt.
         """
 
         if not self.messages:
            return "No previous conversation."

         formatted_history = []

         for message in self.messages:

          formatted_history.append(
            f"{message.role.capitalize()}:\n{message.content}"
        )

         return "\n\n".join(formatted_history)
 
