
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
        with st.expander("View documents"):
            for f in unique_files:
                st.caption(f"{f}")

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
# st.title("Local Document Assistant")
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