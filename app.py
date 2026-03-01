
import streamlit as st
from rag_engine import RAGEngine


# initialize the rag engine. cached so it only loads once per session
@st.cache_resource
def load_engine():
    """Load and cache the RAG engine so documents aren't re-indexed on every rerender."""
    return RAGEngine()

engine = load_engine()

st.title("Local Document Assistant")
st.caption("Currently powered by Ollama + LlamaIndex + ChromaDB")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the RAG engine and display the response
    with st.chat_message("assistant"):
        response = engine.query(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})