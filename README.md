Streamlit Link: https://document-rag-agent-01.streamlit.app/

# RAG Agent – Upload Document (PDF, Word, Excel) and ask questions about it.
# Stack: OpenAI API + LangChain + FAISS + Streamlit
# 📄 Document RAG Agent (Multi-Format Support)

An advanced Retrieval-Augmented Generation (RAG) web application built using Streamlit, LangChain, and OpenAI. The application allows users to upload documents in multiple formats (`.pdf`, `.docx`, `.doc`, `.xlsx`, `.xls`, `.txt`) and chat with them in real-time. Answers are generated strictly using the content extracted from the document, along with full source transparency and page citations.

---

## ✨ Features & Key Highlights

* Multi-Format Processing: Seamlessly handles PDF, Word, Excel, and Text documents via dynamic loader selection.
* Ground-Truth Question Answering: Enforces strict system prompts so the LLM responds only using context provided by the document.
* Transparent Citations: Built-in chunk viewer via Streamlit Expanders allowing users to inspect exact source chunks and page numbers.
* Smart Caching (`file_sig`): Efficiently reuses the FAISS vector index in `st.session_state` unless the uploaded document or chunk parameters change.
* Configurable RAG Pipeline: Adjust Chunk Size, Chunk Overlap, Top-K retrieval, and Model choices (`gpt-4o-mini`, `gpt-4o`, etc.) live from the sidebar.
* Clean UI/UX: Lightweight, responsive interface tailored for seamless document interactions.

---

## 🛠️ Tech Stack & Dependencies

* Frontend Framework: Streamlit
* LLM Orchestration: LangChain (`langchain-core`, `langchain-community`, `langchain-openai`)
* Vector Database: FAISS (`faiss-cpu`)
* Embeddings: OpenAI `text-embedding-3-small`
* File Parsers: `pypdf`, `python-docx`, `openpyxl`, `unstructured`, `msoffcrypto-tool`

---

## 📂 Project Architecture

```text
├── app.py              # Main Streamlit application entry point
├── requirements.txt    # Project dependencies list
└── README.md           # Documentation