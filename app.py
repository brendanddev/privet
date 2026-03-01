
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

# initialize chat histoy in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("Ask a question about your documents..."):
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

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})