import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import chromadb

# llm and embedding models
Settings.llm = Ollama(model="gemma3:1b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# chroma db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load docxs
docs = SimpleDirectoryReader("docs").load_data()
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
query_engine = index.as_query_engine()

st.title("Local Document Assistant")
st.caption("Powered by Ollama + LlamaIndex + ChromaDB")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        st.markdown(str(response))
        st.session_state.messages.append({"role": "assistant", "content": str(response)})