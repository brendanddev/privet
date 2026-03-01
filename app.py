
import os
import streamlit as st
from rag_engine import RAGEngine
from logger import setup_logger
import chromadb

logger = setup_logger()

@st.cache_resource
def load_engine():
    """Load and cache the RAG engine so documents aren't re-indexed on every rerender."""
    logger.info("Loading RAG engine via Streamlit cache")
    return RAGEngine()

def get_collection_stats():
    """Fetch chunk and document count directly from ChromaDB."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("documents")
    results = collection.get(include=["metadatas"])
    total_chunks = collection.count()
    unique_files = set(m.get("file_name", "unknown") for m in results["metadatas"])
    return total_chunks, unique_files

engine = load_engine()
logger.info("Streamlit app started")

# Sidebar
with st.sidebar:
    st.title("Debug Panel")
    st.divider()

    st.subheader("Engine Stats")
    stats = engine.get_stats()
    st.metric("Startup Time", f"{stats['startup_time']}s")
    st.metric("Last Query Time", f"{stats['last_query_time']}s" if stats['last_query_time'] else "No queries yet")
    st.caption(f"**LLM:** {stats['llm_model']}")
    st.caption(f"**Embed:** {stats['embed_model']}")

    st.divider()

    st.subheader("Collection Stats")
    total_chunks, unique_files = get_collection_stats()
    st.metric("Total Chunks", total_chunks)
    st.metric("Documents Loaded", len(unique_files))

    if unique_files:
        with st.expander("Manage documents"):
            for f in unique_files:
                col1, col2 = st.columns([3, 1])
                exists_on_disk = os.path.exists(os.path.join("docs", f))
                with col1:
                    if exists_on_disk:
                        st.caption(f"{f}")
                    else:
                        st.caption(f"{f} _(not in docs folder)_")
                with col2:
                    if st.button("Remove", key=f"remove_{f}"):
                        success = engine.remove_document(f)
                        if success:
                            file_path = os.path.join("docs", f)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            st.success(f"Removed {f}")
                            logger.info(f"Document removed via UI: {f}")
                            st.rerun()
                        else:
                            st.error(f"Could not remove {f}")

    st.divider()

    st.subheader("Last Query Sources")
    if "last_sources" not in st.session_state:
        st.caption("No queries yet.")
    else:
        for i, source in enumerate(st.session_state.last_sources):
            st.caption(f"**[{i+1}] {source['file']} — Page {source['page']}**")
            if source["score"]:
                st.progress(min(float(source["score"]), 1.0), text=f"Score: {source['score']}")
            st.caption(f"_{source['preview']}_")
            st.divider()


# Main section
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
        # Save the uploaded file to the docs folder
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    logger.info(f"User submitted prompt via UI: '{prompt}'")

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = engine.query(prompt)
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