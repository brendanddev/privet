
import os
import time
import chromadb
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from core.embeddings import Float16EmbeddingWrapper
from utils.logger import setup_logger
from utils.config import load_config

class RAGEngine:
    """
    Handles all RAG pipeline logic including model configuration, document ingestion,
    vector storage, and query execution.

    This class is intentionally decoupled from the UI layer so it can be reused, tested,
    or swapped out independently.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the RAG engine from a config file.

        Args:
            config_path (str): Path to the YAML config file
        """
        config = load_config(config_path)

        self.docs_path = config["docs_path"]
        self.chroma_path = config["chroma_path"]
        self.collection_name = config["collection_name"]
        self.llm_model = config["llm_model"]
        self.embed_model_name = config["embed_model"]
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.similarity_top_k = config["similarity_top_k"]
        self.history_length = config["history_length"]
        self.startup_time = None
        self.last_query_time = None
        self.last_sources = []

        self.logger = setup_logger()
        self.logger.info(f"Initializing RAGEngine | LLM: {self.llm_model} | Embed: {self.embed_model_name}")

        # OLLAMA_HOST env var takes priority over config file
        # This allows Docker to override without changing config.yaml
        ollama_host = os.environ.get("OLLAMA_HOST", config["ollama_host"])

        Settings.llm = Ollama(
            model=self.llm_model,
            request_timeout=config["request_timeout"],
            base_url=ollama_host
        )
        base_embed = OllamaEmbedding(
            model_name=self.embed_model_name,
            base_url=ollama_host
        )
        Settings.embed_model = Float16EmbeddingWrapper(base_embed)

        start = time.time()
        self.query_engine = self._build_query_engine()
        self.startup_time = round(time.time() - start, 2)

        self.logger.info(f"Engine ready | Startup time: {self.startup_time}s | Chunks indexed: {self._get_chunk_count()}")

    def _build_query_engine(self):
        """
        Sets up ChromaDB, ingests documents, and builds the index.

        Checks if documents have already been indexed and skips re-embedding if so.
        This dramatically reduces startup time on subsequent runs.

        If the docs folder is empty and no existing index exists, returns a query
        engine over an empty index rather than crashing. Documents can be added
        through the UI without restarting.

        Returns:
            query_engine: A LlamaIndex query engine ready to answer questions
        """
        self.logger.info(f"Building query engine | Docs path: {self.docs_path} | Collection: {self.collection_name}")

        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        existing_count = chroma_collection.count()

        if existing_count > 0:
            self.logger.info(f"Existing index found | {existing_count} chunks | Skipping re-indexing")
            index = VectorStoreIndex.from_vector_store(vector_store)

        else:
            # Check if docs folder has any files before attempting to load
            os.makedirs(self.docs_path, exist_ok=True)
            doc_files = [
                f for f in os.listdir(self.docs_path)
                if not f.startswith(".")
            ]

            if not doc_files:
                # No documents and no existing index — build an empty index
                # The app will still load and the user can upload documents via the UI
                self.logger.warning("No documents found in docs/ folder — starting with empty index")
                index = VectorStoreIndex.from_vector_store(vector_store)
            else:
                self.logger.info("No existing index found — indexing documents for the first time")

                splitter = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

                docs = SimpleDirectoryReader(self.docs_path).load_data()
                self.logger.info(f"Loaded {len(docs)} document pages from {self.docs_path}")

                index = VectorStoreIndex.from_documents(
                    docs,
                    storage_context=storage_context,
                    transformations=[splitter]
                )
                self.logger.info("Vector index built and persisted to ChromaDB")

        self.streaming_query_engine = index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            streaming=True
        )
        return index.as_query_engine(similarity_top_k=self.similarity_top_k)

    def _get_chunk_count(self) -> int:
        """
        Return total number of chunks stored in the collection.

        Returns:
            int: Total number of chunks in the ChromaDB collection
        """
        client = chromadb.PersistentClient(path=self.chroma_path)
        collection = client.get_or_create_collection(self.collection_name)
        return collection.count()

    def _build_history_context(self, chat_history: list) -> str:
        """
        Build a context string from recent chat history.

        Extracted into its own method to avoid duplicating this logic
        across query() and stream_query().

        Args:
            chat_history (list): List of dicts with 'role' and 'content' keys

        Returns:
            str: Formatted history string, empty string if no history
        """
        if not chat_history:
            return ""

        recent = chat_history[-self.history_length:]
        context = ""
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        return context

    def _build_contextual_question(self, question: str, history_context: str) -> str:
        """
        Prepend conversation history to the question for the model.

        Args:
            question (str): The user's current question
            history_context (str): Formatted history string from _build_history_context

        Returns:
            str: Full prompt with history and question
        """
        if not history_context:
            return question

        return f"""Previous conversation:
{history_context}
Current question: {question}

Answer the current question, taking into account the conversation above if relevant."""

    def query(self, question: str, chat_history: list = None) -> dict:
        """
        Run a question through the RAG pipeline with optional conversation history.

        Args:
            question (str): The user's question
            chat_history (list): List of dicts with 'role' and 'content' keys

        Returns:
            dict: Contains 'answer' (str), 'sources' (list of dicts), and 'query_time' (float)
        """
        self.logger.info(f"Query received: '{question}'")

        history_context = self._build_history_context(chat_history)
        contextual_question = self._build_contextual_question(question, history_context)

        start = time.time()
        response = self.query_engine.query(contextual_question)
        self.last_query_time = round(time.time() - start, 2)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "file": node.metadata.get("file_name", "Unknown"),
                "page": node.metadata.get("page_label", "Unknown"),
                "score": round(node.score, 4) if node.score else None,
                "preview": node.text[:150]
            })

        self.last_sources = sources
        self.logger.info(f"Query completed | Time: {self.last_query_time}s | Sources retrieved: {len(sources)}")

        return {
            "answer": str(response),
            "sources": sources,
            "query_time": self.last_query_time
        }

    def stream_query(self, question: str, chat_history: list = None):
        """
        Run a question through the RAG pipeline and stream the response token by token.

        Retrieval happens upfront, source nodes are available immediately after the
        streaming response object is created, before any tokens are generated. This is
        how LlamaIndex's streaming response works internally: the retrieval step is
        synchronous, only the generation step is streamed. We extract sources from
        source_nodes right after retrieval and store them on self.last_sources so
        app.py can access them without running a second query.

        Args:
            question (str): The user's question
            chat_history (list): List of dicts with 'role' and 'content' keys

        Yields:
            str: Each token as it is generated
        """
        self.logger.info(f"Stream query received: '{question}'")

        history_context = self._build_history_context(chat_history)
        contextual_question = self._build_contextual_question(question, history_context)

        start = time.time()

        # Retrieval happens here synchronously, source_nodes are populated
        # before any tokens are generated
        streaming_response = self.streaming_query_engine.query(contextual_question)

        # Extract sources immediately after retrieval, before streaming begins
        # This is why we don't need a second query call in app.py
        self.last_sources = []
        for node in streaming_response.source_nodes:
            self.last_sources.append({
                "file": node.metadata.get("file_name", "Unknown"),
                "page": node.metadata.get("page_label", "Unknown"),
                "score": round(node.score, 4) if node.score else None,
                "preview": node.text[:150]
            })

        self.logger.info(f"Sources retrieved: {len(self.last_sources)} | Starting token stream")

        # Now stream the generation step token by token
        for token in streaming_response.response_gen:
            yield token

        self.last_query_time = round(time.time() - start, 2)
        self.logger.info(f"Stream query completed | Time: {self.last_query_time}s")

    def get_stats(self) -> dict:
        """
        Return current engine performance stats.

        Returns:
            dict: Startup time, last query time, model names
        """
        return {
            "startup_time": self.startup_time,
            "last_query_time": self.last_query_time,
            "llm_model": self.llm_model,
            "embed_model": self.embed_model_name,
        }

    def add_document(self, file_path: str) -> int:
        """
        Ingest a single new document and add it to the existing index.

        Does not re-index already embedded documents, only processes the new file.

        Args:
            file_path (str): Full path to the document to ingest

        Returns:
            int: Number of new chunks added to the index
        """
        self.logger.info(f"Adding new document: {file_path}")

        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        self.logger.info(f"Loaded {len(docs)} pages from {file_path}")

        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        chunks_before = chroma_collection.count()

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store)

        for doc in docs:
            index.insert(doc)

        chunks_after = chroma_collection.count()
        new_chunks = chunks_after - chunks_before

        self.query_engine = index.as_query_engine(similarity_top_k=self.similarity_top_k)
        self.streaming_query_engine = index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            streaming=True
        )

        self.logger.info(f"Document added | New chunks: {new_chunks} | Total chunks: {chunks_after}")
        return new_chunks

    def remove_document(self, file_name: str) -> bool:
        """
        Remove all chunks belonging to a specific document from the vector store.

        Args:
            file_name (str): The filename to remove (ex: 'test.pdf')

        Returns:
            bool: True if removed successfully, False if not found
        """
        self.logger.info(f"Removing document: {file_name}")

        client = chromadb.PersistentClient(path=self.chroma_path)
        collection = client.get_or_create_collection(self.collection_name)

        results = collection.get(include=["metadatas"])
        ids_to_delete = [
            results["ids"][i]
            for i, m in enumerate(results["metadatas"])
            if m.get("file_name") == file_name
        ]

        if not ids_to_delete:
            self.logger.warning(f"No chunks found for {file_name}")
            return False

        collection.delete(ids=ids_to_delete)
        self.logger.info(f"Removed {len(ids_to_delete)} chunks for {file_name}")

        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        self.query_engine = index.as_query_engine(similarity_top_k=self.similarity_top_k)
        self.streaming_query_engine = index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            streaming=True
        )

        return True

    def switch_models(self, llm_model: str, embed_model: str):
        """
        Switch the LLM and embedding model without restarting the app.

        Args:
            llm_model (str): New Ollama LLM model name
            embed_model (str): New Ollama embedding model name
        """
        self.logger.info(f"Switching models | LLM: {llm_model} | Embed: {embed_model}")

        self.llm_model = llm_model
        self.embed_model_name = embed_model

        Settings.llm = Ollama(model=llm_model, request_timeout=120.0)
        base_embed = OllamaEmbedding(model_name=embed_model)
        Settings.embed_model = Float16EmbeddingWrapper(base_embed)

        self.query_engine = self._build_query_engine()
        self.logger.info(f"Model switch complete | LLM: {llm_model} | Embed: {embed_model}")