# 📄 RAG Document Chatbot

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

A Retrieval-Augmented Generation (RAG) chatbot designed to intelligently interact with your PDF documents. By leveraging Large Language Models (LLMs) and vector databases, this application transforms static documents into dynamic, conversational interfaces. 

Users can upload documents, and the system seamlessly extracts text, generates dense vector embeddings, and retrieves highly relevant context to provide accurate, context-aware answers to user queries.

---

## ✨ Key Features

* **PDF Processing:** Efficiently upload, parse, and extract text from complex PDF documents.
* **Semantic Search:** Generates and stores high-dimensional embeddings for accurate semantic document retrieval.
* **Context-Aware Responses:** Synthesizes retrieved document chunks with LLM generation for precise answers.
* **Conversation Memory:** Maintains chat history for natural, multi-turn interactions.
* **Modular Architecture:** Clean, separated code structure allowing for easy scaling and component swapping.

---

## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python |
| **Vector Database** | ChromaDB |
| **Embeddings** | Hugging Face |
| **LLM Orchestration** | LangChain |
| **Document Parsing** | PyPDF |
| **Frontend Framework** | Streamlit |

---

## 📂 Project Structure

```text
rag-document-chatbot/
├── app.py                # Main application entry point
├── embeddings/           # Embedding generation and model configuration
├── frontend/             # UI components and layout rendering
├── llm/                  # Large Language Model integration and prompting
├── loaders/              # Document ingestion and parsing utilities
├── memory/               # Conversation history and state management
├── models/               # Data schemas and model definitions
├── rag/                  # Core Retrieval-Augmented Generation logic
├── utils/                # Helper functions and text pre-processing
└── vectorstore/          # ChromaDB connection and indexing

🚀 Getting Started
Prerequisites
Ensure you have the following installed on your local machine:

Python 3.8 or higher

Git

Installation
1. Clone the repository to your local machine:

git clone <repository-url>
cd rag-document-chatbot

2. Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:

pip install -r requirements.txt

Note: Make sure to set up your environment variables (like API keys) in a .env file before running the application.

▶️ Usage
Start the application by running the main entry file:

python app.py

Once the server is running, navigate to the provided localhost URL in your browser to start uploading documents and chatting.

📌 Roadmap & Future Improvements
[ ] Multi-Document Support: Query across an entire knowledge base of multiple PDFs simultaneously.

[ ] History-Aware Query Rewriting: Optimize user follow-up questions for better vector retrieval.

[ ] Streaming Responses: Implement real-time token streaming for a faster user experience.

[ ] Dockerization: Containerize the application for seamless deployment.

[ ] Advanced Chunking: Implement semantic or recursive character chunking for better context window management.

👨‍💻 Author
Mahi Yadav * GitHub: @YourGitHubUsername

LinkedIn: Mahi Yadav

If you found this project helpful, please consider giving it a ⭐ on GitHub!