
import os
import streamlit as st
import chromadb
from core.rag_engine import RAGEngine
from ui.dashboard import render_sidebar
from utils.logger import setup_logger
from utils.feedback import log_feedback
from utils.config import load_config

logger = setup_logger()


@st.cache_resource
def load_engine():
    """Load and cache the RAG engine so documents are not re-indexed on every rerender."""
    logger.info("Loading RAG engine via Streamlit cache")
    return RAGEngine()


@st.cache_resource
def get_config():
    """
    Load and cache config so it doesn't reload on every Streamlit rerun.

    Without caching, load_config() fires on every single rerender which
    happens constantly in Streamlit — on every button click, input change,
    or state update. Caching it once means one file read for the session.

    Returns:
        dict: App configuration
    """
    return load_config()


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
config = get_config()
logger.info("Streamlit app started")

render_sidebar(engine, get_collection_stats)

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

# Initialize all session state upfront so nothing is ever missing on rerun
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    logger.info(f"User submitted prompt via UI: '{prompt}'")

    # Reset feedback state for the new question before anything else
    # This must happen here — tied to a new prompt being submitted,
    # not to a rerun, so the buttons reappear for every new answer
    st.session_state.feedback_given = False
    st.session_state.last_prompt = prompt

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for token in engine.stream_query(prompt, chat_history=st.session_state.messages):
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        st.caption(f"Response time: {engine.last_query_time}s")

        # Sources already populated from streaming — no second query needed
        sources = engine.last_sources
        st.session_state.last_sources = sources
        st.session_state.last_response = full_response

        if sources:
            with st.expander("Sources"):
                for i, source in enumerate(sources):
                    st.markdown(f"**[{i+1}] {source['file']} — Page {source['page']}**")
                    if source["score"]:
                        st.caption(f"Relevance score: {source['score']}")
                    st.caption(f"_{source['preview']}_")
                    st.divider()

        logger.info(f"Response delivered | Answer length: {len(full_response)} chars | Query time: {engine.last_query_time}s")
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Feedback buttons — outside the prompt block so they persist across reruns
# Reads from session state which is always populated after first question
if config.get("collect_feedback", True) and st.session_state.last_prompt:
    if not st.session_state.feedback_given:
        st.caption("Was this answer helpful?")
        col1, col2, col3 = st.columns([1, 1, 8])

        with col1:
            if st.button("👍", key="thumbs_up"):
                log_feedback(
                    feedback_path=config["feedback_path"],
                    question=st.session_state.last_prompt,
                    answer=st.session_state.last_response,
                    rating="thumbs_up",
                    sources=st.session_state.last_sources,
                    model=engine.llm_model,
                    query_time=engine.last_query_time
                )
                st.session_state.feedback_given = True
                st.toast("Thanks for the feedback!")
                st.rerun()

        with col2:
            if st.button("👎", key="thumbs_down"):
                log_feedback(
                    feedback_path=config["feedback_path"],
                    question=st.session_state.last_prompt,
                    answer=st.session_state.last_response,
                    rating="thumbs_down",
                    sources=st.session_state.last_sources,
                    model=engine.llm_model,
                    query_time=engine.last_query_time
                )
                st.session_state.feedback_given = True
                st.toast("Got it — noted for improvement.")
                st.rerun()
    else:
        st.caption("✓ Feedback received")