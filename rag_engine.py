
import time
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from logger import setup_logger


class RAGEngine:
    """
    Handles all RAG pipeline logic including model configuration, document ingestion,
    vector storage, and query execution.

    This class is intentionally decoupled from the UI layer so it can be reused, tested,
    or swapped out independently.
    """

    def __init__(
        self,
        docs_path: str = "./docs",
        chroma_path: str = "./chroma_db",
        collection_name: str = "documents",
        llm_model: str = "gemma3:1b",
        embed_model: str = "nomic-embed-text",
        request_timeout: float = 120.0
    ):
        """
        Initialize the RAG engine and build the query engine from documents.

        Args:
            docs_path (str): Path to the folder containing documents to index
            chroma_path (str): Path to the ChromaDB persistence directory
            collection_name (str): ChromaDB collection name
            llm_model (str): Ollama LLM model name
            embed_model (str): Ollama embedding model name
            request_timeout (float): Timeout in seconds for Ollama requests
        """
        self.docs_path = docs_path
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.embed_model_name = embed_model
        self.startup_time = None
        self.last_query_time = None

        # Set up logger for the engine
        self.logger = setup_logger()
        self.logger.info(f"Initializing RAGEngine | LLM: {llm_model} | Embed: {embed_model}")

        # Configure global LlamaIndex settings with local Ollama models
        Settings.llm = Ollama(model=llm_model, request_timeout=request_timeout)
        Settings.embed_model = OllamaEmbedding(model_name=embed_model)

        # Build the query engine and record how long it takes
        start = time.time()
        self.query_engine = self._build_query_engine()
        self.startup_time = round(time.time() - start, 2)

        self.logger.info(f"Engine ready | Startup time: {self.startup_time}s | Chunks indexed: {self._get_chunk_count()}")

    def _build_query_engine(self):
        """
        Internal method to set up ChromaDB, ingest documents, and build the index.

        Checks if documents have already been indexed and skips re-embedding if so.
        This dramatically reduces startup time on subsequent runs.

        Returns:
            query_engine: A LlamaIndex query engine ready to answer questions
        """
        self.logger.info(f"Building query engine | Docs path: {self.docs_path} | Collection: {self.collection_name}")

        # Connect to ChromaDB and get or create the collection
        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)

        # Wrap the ChromaDB collection as a LlamaIndex vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Check if the collection already has documents indexed
        existing_count = chroma_collection.count()

        if existing_count > 0:
            # Index already exists — load it directly without re-embedding
            self.logger.info(f"Existing index found | {existing_count} chunks | Skipping re-indexing")
            index = VectorStoreIndex.from_vector_store(vector_store)
        else:
            # No index found — load documents and embed them for the first time
            self.logger.info("No existing index found — indexing documents for the first time")
            docs = SimpleDirectoryReader(self.docs_path).load_data()
            self.logger.info(f"Loaded {len(docs)} document pages from {self.docs_path}")
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
            self.logger.info("Vector index built and persisted to ChromaDB")

        return index.as_query_engine(similarity_top_k=3)

    def _get_chunk_count(self) -> int:
        """
        Return total number of chunks stored in the collection.

        Used for logging and benchmarking after indexing completes.

        Returns:
            int: Total number of chunks in the ChromaDB collection
        """
        client = chromadb.PersistentClient(path=self.chroma_path)
        collection = client.get_or_create_collection(self.collection_name)
        return collection.count()

    def query(self, question: str) -> dict:
        """
        Run a question through the RAG pipeline and return the answer with sources.

        Args:
            question (str): The user's question

        Returns:
            dict: Contains 'answer' (str), 'sources' (list of dicts), and 'query_time' (float)
        """
        self.logger.info(f"Query received: '{question}'")

        # Time the query so we can benchmark response speed
        start = time.time()
        response = self.query_engine.query(question)
        self.last_query_time = round(time.time() - start, 2)

        # Extract source information from the retrieved chunks
        sources = []
        for node in response.source_nodes:
            sources.append({
                "file": node.metadata.get("file_name", "Unknown"),
                "page": node.metadata.get("page_label", "Unknown"),
                "score": round(node.score, 4) if node.score else None,
                "preview": node.text[:150]
            })

        self.logger.info(f"Query completed | Time: {self.last_query_time}s | Sources retrieved: {len(sources)}")

        return {
            "answer": str(response),
            "sources": sources,
            "query_time": self.last_query_time
        }

    def get_stats(self) -> dict:
        """
        Return current engine performance stats.

        Useful for displaying benchmarks in the UI or logging baseline metrics.

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

        Does not re-index already embedded documents — only processes the new file.

        Args:
            file_path (str): Full path to the document to ingest

        Returns:
            int: Number of new chunks added to the index
        """
        self.logger.info(f"Adding new document: {file_path}")

        # Load only the new document
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        self.logger.info(f"Loaded {len(docs)} pages from {file_path}")

        # Connect to existing ChromaDB collection
        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        chunks_before = chroma_collection.count()

        # Add new document to the existing vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        for doc in docs:
            index.insert(doc)

        chunks_after = chroma_collection.count()
        new_chunks = chunks_after - chunks_before

        # Refresh the query engine so it includes the new document
        self.query_engine = index.as_query_engine(similarity_top_k=3)

        self.logger.info(f"Document added | New chunks: {new_chunks} | Total chunks: {chunks_after}")
        return new_chunks