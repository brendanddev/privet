
import streamlit as st
from rag_engine import RAGEngine
from logger import setup_logger

logger = setup_logger()

@st.cache_resource
def load_engine():
    """Load and cache the RAG engine so documents aren't re-indexed on every rerender."""
    logger.info("Loading RAG engine via Streamlit cache")
    return RAGEngine()

engine = load_engine()
logger.info("Streamlit app started")

st.title("Local Document Assistant")
st.caption("Powered by Ollama + LlamaIndex + ChromaDB")

# Privacy status bar
st.success("Fully local — no data leaves your machine. No cloud, no API calls, no tracking.")

with st.expander("How this works"):
    st.markdown("""
    - Your documents are processed and stored **entirely on your machine**
    - The AI model runs locally via **Ollama** — no OpenAI, no external APIs
    - Embeddings and document chunks are stored in a **local ChromaDB database**
    - Nothing is transmitted to any external server at any point
    """)

st.divider()

# Initialize chat history in session state
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