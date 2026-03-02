
import os
import streamlit as st
import chromadb
from core.rag_engine import RAGEngine
from ui.dashboard import render_sidebar
from utils.logger import setup_logger

logger = setup_logger()


@st.cache_resource
def load_engine():
    """Load and cache the RAG engine so documents are not re-indexed on every rerender."""
    logger.info("Loading RAG engine via Streamlit cache")
    return RAGEngine()


def get_collection_stats():
    """
    Fetch chunk count and unique document names directly from ChromaDB.

    Returns:
        tuple: (total_chunks int, unique_files set)
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("documents")
    results = collection.get(include=["metadatas"])
    total_chunks = collection.count()
    unique_files = set(m.get("file_name", "unknown") for m in results["metadatas"])
    return total_chunks, unique_files


engine = load_engine()
logger.info("Streamlit app started")

# Render sidebar debug panel
render_sidebar(engine, get_collection_stats)

# Main UI
st.title("Local Document Assistant")
st.caption("Powered by Ollama + LlamaIndex + ChromaDB")

st.success("Fully local — no data leaves your machine. No cloud, no API calls, no tracking.")

with st.expander("How this works"):
    st.markdown("""
    - Your documents are processed and stored **entirely on your machine**
    - The AI model runs locally via **Ollama** — no OpenAI, no external APIs
    - Embeddings and document chunks are stored in a **local ChromaDB database**
    - Nothing is transmitted to any external server at any point
    """)

st.divider()

# Document upload
with st.expander("Upload a document"):
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx", "csv"],
        help="Your file will be saved locally and added to the index"
    )

    if uploaded_file is not None:
        save_path = os.path.join("docs", uploaded_file.name)

        if os.path.exists(save_path):
            st.warning(f"{uploaded_file.name} is already in your docs folder.")
        else:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Indexing {uploaded_file.name}..."):
                new_chunks = engine.add_document(save_path)

            st.success(f"{uploaded_file.name} added — {new_chunks} new chunks indexed")
            logger.info(f"File uploaded via UI: {uploaded_file.name} | Chunks added: {new_chunks}")

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    logger.info(f"User submitted prompt via UI: '{prompt}'")

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = engine.query(prompt, chat_history=st.session_state.messages)
        st.markdown(result["answer"])
        st.caption(f"Response time: {result['query_time']}s")

        # Store sources in session state so sidebar can display them
        st.session_state.last_sources = result["sources"]

        if result["sources"]:
            with st.expander("Sources"):
                for i, source in enumerate(result["sources"]):
                    st.markdown(f"**[{i+1}] {source['file']} — Page {source['page']}**")
                    if source["score"]:
                        st.caption(f"Relevance score: {source['score']}")
                    st.caption(f"_{source['preview']}_")
                    st.divider()

        logger.info(f"Response delivered | Answer length: {len(result['answer'])} chars | Query time: {result['query_time']}s")
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})