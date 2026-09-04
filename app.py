# RAG Agent – Upload Document (PDF, Word, Excel) and ask questions about it.
# Stack: OpenAI API + LangChain + FAISS + Streamlit

import os
import tempfile
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# Page setup & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Document RAG Agent", page_icon="📄", layout="wide")

# Light Clean Theme Styling
st.markdown(
    """
    <style>
    /* Main Background & Font */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Title Accent */
    h1 {
        color: #1E293B;
        font-weight: 700;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 10px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #DCD0FF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Input Fields & Buttons */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        font-weight: 500;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    
    /* Sources Box */
    .streamlit-expanderHeader {
        background-color: #EDF2F7 !important;
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Document RAG Agent")
st.caption("Upload a file, then ask questions - answers come only from the document.")

# -----------------------------------------------------------------------------
# Sidebar: API key + settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model_name = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    top_k = st.slider("Retrieved chunks (k)", 2, 10, 4)

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------------------------------------------------------
# File upload + indexing (cached in session state)
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a Document", type=["pdf", "docx", "doc", "xlsx", "xls", "txt"]
)


def build_vectorstore(file_bytes: bytes, file_name: str, size: int, overlap: int) -> FAISS:
    """Load document dynamically by extension, split into chunks, embed, and store in FAISS."""
    ext = os.path.splitext(file_name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp_path)
            content = df.to_string(index=False)
            docs = [Document(page_content=content, metadata={"source": file_name})]
        else:
            loader = TextLoader(tmp_path)
            docs = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


if uploaded_file is not None:
    file_sig = (uploaded_file.name, uploaded_file.size, chunk_size, chunk_overlap)

    if st.session_state.get("file_sig") != file_sig:
        with st.spinner("Reading and indexing document..."):
            st.session_state.vectorstore = build_vectorstore(
                uploaded_file.getvalue(),
                uploaded_file.name,
                chunk_size,
                chunk_overlap,
            )
            st.session_state.file_sig = file_sig
            st.session_state.messages = []
    st.success(f"Indexed **{uploaded_file.name}** and ask away!")

# -----------------------------------------------------------------------------
# RAG chain
# -----------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions strictly using the "
        "provided context from an uploaded document.\n"
        "Rules:\n"
        "1. Answer ONLY from the context below.\n"
        "2. If the answer is not in the context, say: "
        "\"I couldn't find that in the document.\"\n"
        "3. Cite source details or page numbers if available.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    """Join retrieved chunks, tagging each with its page/source number."""
    return "\n\n".join(
        f"[Source Info: Page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def get_chain(vectorstore: FAISS, model: str, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0)
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ), retriever


# -----------------------------------------------------------------------------
# Chat Interface
# -----------------------------------------------------------------------------
if "vectorstore" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        chain, retriever = get_chain(
            st.session_state.vectorstore, model_name, top_k
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
                st.markdown(answer)

            # Show the retrieved chunks for transparency
            with st.expander("🔍 Sources (retrieved chunks)"):
                for doc in retriever.invoke(question):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page + 1 if isinstance(page, int) else page}**")
                    st.text(doc.page_content[:500])
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload a File to get started.")