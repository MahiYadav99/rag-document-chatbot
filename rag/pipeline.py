from llm.gemini_client import GeminiClient
from llm.response_generator import generate_rag_response
from memory.conversation_memory import ConversationMemory
from rag.prompt import PromptType


class RAGPipeline:
    """
    Orchestrates the complete RAG workflow.
    """

    def __init__(
        self,
        retriever,
        gemini_client: GeminiClient,
        memory: ConversationMemory,
    ):
        self.retriever = retriever
        self.gemini_client = gemini_client
        self.memory = memory

    def ask(
        self,
        question: str,
        prompt_type: PromptType = PromptType.QA,
    ):

        print("\n" + "=" * 70)
        print("NEW QUESTION")
        print("=" * 70)
        print(f"Question: {question}")

        # -------------------------------
        # Memory BEFORE asking
        # -------------------------------
        print("\nMEMORY BEFORE")
        print("-" * 70)
        print(self.memory.format_history())

        # -------------------------------
        # Retrieve documents
        # -------------------------------
        print("\nRETRIEVING DOCUMENTS...")
        documents = self.retriever.invoke(question)

        print(f"Retrieved {len(documents)} document(s).\n")

        for i, doc in enumerate(documents, start=1):
            print(f"Document {i}")
            print(f"Page : {doc.metadata.get('page_label', doc.metadata.get('page'))}")
            print(f"Source : {doc.metadata.get('source')}")
            print(f"Preview : {doc.page_content[:120]}...")
            print("-" * 70)

        # -------------------------------
        # Conversation history
        # -------------------------------
        history = self.memory.format_history()

        print("\nHISTORY SENT TO GEMINI")
        print("-" * 70)
        print(history)

        # -------------------------------
        # Generate response
        # -------------------------------
        print("\nGENERATING RESPONSE...")
        response = generate_rag_response(
            question=question,
            documents=documents,
            conversation_history=history,
            prompt_type=prompt_type,
            gemini_client=self.gemini_client,
        )

        print("\nGEMINI RESPONSE")
        print("-" * 70)
        print(response.answer)

        # -------------------------------
        # Save conversation
        # -------------------------------
        print("\nSAVING USER MESSAGE...")
        self.memory.add_user_message(question)

        print(f"Messages after USER: {len(self.memory.get_history())}")

        print("\nSAVING ASSISTANT MESSAGE...")
        self.memory.add_assistant_message(response.answer)

        print(f"Messages after ASSISTANT: {len(self.memory.get_history())}")

        # -------------------------------
        # Memory AFTER asking
        # -------------------------------
        print("\nMEMORY AFTER")
        print("-" * 70)
        print(self.memory.format_history())

        print("\nEND OF QUESTION")
        print("=" * 70)

        return response

    def clear_memory(self):
        self.memory.clear()

    def get_conversation_history(self):
        return self.memory.get_history()