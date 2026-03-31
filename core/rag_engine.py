
import os
import time
import chromadb
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from utils.logger import setup_logger
from utils.config import load_config
from core.providers.factory import get_provider


class RAGEngine:
    """
    Handles all RAG pipeline logic including model configuration, document ingestion,
    vector storage, and query execution.

    This class is intentionally decoupled from the UI layer so it can be reused, tested,
    or swapped out independently. The provider abstraction means the engine works with
    any backend, Ollama, llama.cpp, or future providers, without changes to this class.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the RAG engine from a config file.

        Args:
            config_path (str): Path to the YAML config file
        """
        config = load_config(config_path)

        self.config = config
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

        # Single ChromaDB client for the lifetime of this engine instance.
        # All methods share this client.... no more per-operation PersistentClient().
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.logger.info(f"ChromaDB client opened | Path: {self.chroma_path}")

        # Load provider from config: llama or llamacpp
        self.provider = get_provider(config)

        # Wire provider into LlamaIndex settings
        Settings.llm = self.provider.llm if hasattr(self.provider, 'llm') else Settings.llm
        Settings.embed_model = self.provider.embed_model

        start = time.time()
        self.query_engine = self._build_query_engine()
        self.startup_time = round(time.time() - start, 2)

        self.logger.info(f"Engine ready | Startup time: {self.startup_time}s | Chunks indexed: {self._get_chunk_count()}")

    def _get_collection(self) -> chromadb.Collection:
        """
        Return the ChromaDB collection, creating it if it does not exist.

        Uses the shared self.chroma_client, never opens a new client.

        Returns:
            chromadb.Collection: The active document collection
        """
        return self.chroma_client.get_or_create_collection(self.collection_name)

    def _get_chunk_count(self) -> int:
        """
        Return total number of chunks stored in the collection.

        Returns:
            int: Total chunk count
        """
        return self._get_collection().count()

    def _build_vector_store_and_index(self) -> tuple:
        """
        Build a ChromaVectorStore and VectorStoreIndex from the shared client.

        Extracted so add_document, remove_document, and _build_query_engine
        don't each inline the same three lines.

        Returns:
            tuple: (ChromaVectorStore, VectorStoreIndex)
        """
        collection = self._get_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        return vector_store, index

    def _apply_llamacpp_prompt(self):
        """
        Apply the custom QA prompt template to both query engines when using llamacpp.

        The prompt prevents raw context chunks from leaking into model responses.
        Ollama handles prompt formatting internally, so this is only needed for llamacpp.

        Called after every operation that rebuilds the query engines:
        _build_query_engine, add_document, remove_document, switch_models.
        """
        if self.config.get("provider") != "llamacpp":
            return

        from core.providers.llamacpp import QA_PROMPT
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": QA_PROMPT}
        )
        self.streaming_query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": QA_PROMPT}
        )
        self.logger.info("Applied custom QA prompt for llamacpp provider")

    def _build_engines_from_index(self, index: VectorStoreIndex):
        """
        Build both query engines from a VectorStoreIndex and apply the llamacpp
        prompt if needed.

        Retrieval path (controlled by use_hybrid_search):
          - true:  BM25 + vector retrieval fused with Reciprocal Rank Fusion
          - false: pure vector search

        Reranking (controlled by use_reranking):
          - true:  cross-encoder/ms-marco-MiniLM-L-6-v2 reranks the retrieved
                   candidates down to rerank_top_k (default 5). Retrieval always
                   fetches 10 candidates so the reranker has enough to choose from.
          - false: retrieved candidates are passed through unchanged

        Args:
            index (VectorStoreIndex): The index to build engines from
        """
        use_reranking = self.config.get("use_reranking", False)
        rerank_top_k = self.config.get("rerank_top_k", 5)

        node_postprocessors = []
        if use_reranking:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            node_postprocessors.append(
                SentenceTransformerRerank(
                    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    top_n=rerank_top_k
                )
            )

        if self.config.get("use_hybrid_search"):
            from llama_index.retrievers.bm25 import BM25Retriever
            from llama_index.core.retrievers import QueryFusionRetriever
            from llama_index.core.query_engine import RetrieverQueryEngine

            vector_retriever = index.as_retriever(similarity_top_k=10)
            collection = self._get_collection()
            results = collection.get(include=["documents", "metadatas"])
            from llama_index.core.schema import TextNode
            nodes = [
                TextNode(text=doc, metadata=meta)
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes, similarity_top_k=10
            )
            retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=self.similarity_top_k,
                num_queries=1,
                mode="reciprocal_rerank"
            )
            self.streaming_query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=node_postprocessors,
                streaming=True
            )
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=node_postprocessors
            )
            mode = "hybrid (BM25 + vector + RRF)"
        else:
            from llama_index.core.query_engine import RetrieverQueryEngine

            retriever = index.as_retriever(similarity_top_k=10 if use_reranking else self.similarity_top_k)
            self.streaming_query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=node_postprocessors,
                streaming=True
            )
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=node_postprocessors
            )
            mode = "vector"

        if use_reranking:
            self.logger.info(f"Retrieval mode: {mode} + rerank (top {rerank_top_k})")
        else:
            self.logger.info(f"Retrieval mode: {mode} only")

        self._apply_llamacpp_prompt()

    def _build_query_engine(self):
        """
        Sets up ChromaDB, ingests documents, and builds the index.

        Checks if documents have already been indexed and skips re-embedding if so.
        This dramatically reduces startup time on subsequent runs.

        If the docs folder is empty and no existing index exists, returns a query
        engine over an empty index rather than crashing. Documents can be added
        through the UI without restarting.

        For llamacpp provider, applies a custom prompt template to prevent the
        raw context chunks from leaking into the model's response.

        Returns:
            query_engine: A LlamaIndex query engine ready to answer questions
        """
        self.logger.info(f"Building query engine | Docs path: {self.docs_path} | Collection: {self.collection_name}")

        collection = self._get_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        existing_count = collection.count()

        if existing_count > 0:
            self.logger.info(f"Existing index found | {existing_count} chunks | Skipping re-indexing")
            index = VectorStoreIndex.from_vector_store(vector_store)

        else:
            os.makedirs(self.docs_path, exist_ok=True)
            doc_files = [
                f for f in os.listdir(self.docs_path)
                if not f.startswith(".")
            ]

            if not doc_files:
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

        self._build_engines_from_index(index)
        return self.query_engine

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

        Conversation history is disabled for llamacpp provider because
        small GGUF models struggle to handle both RAG context and chat
        history simultaneously without hallucinating conversation structure.
        History will be re-enabled once a larger model is tested.

        Args:
            question (str): The user's current question
            history_context (str): Formatted history string from _build_history_context

        Returns:
            str: Full prompt with or without history depending on provider
        """
        # Disable history for llamacpp — small models hallucinate with combined context
        if self.config.get("provider") == "llamacpp":
            return question

        if not history_context:
            return question

        return f"""Previous conversation:
{history_context}
Current question: {question}

Answer the current question, taking into account the conversation above if relevant."""

    def _enrich_scores_from_chroma(self, question: str, sources: list) -> None:
        """
        Inject real cosine similarity scores into source dicts by querying ChromaDB directly.

        LlamaIndex's ChromaVectorStore does not populate NodeWithScore.score in this
        version (0.14.15), so all scores come back as 0.0. This method re-queries
        ChromaDB with the same embedding to get actual cosine distances, converts
        them to similarity scores (score = 1 - distance), and writes them back into
        the source dicts in place.

        Matching is done by file_name from metadata because ChromaDB result ordering
        is not guaranteed to align with LlamaIndex source_nodes ordering.

        Falls back silently — existing 0.0 scores are preserved on any error so the
        query result is never degraded.

        Args:
            question (str): The raw user question to embed
            sources (list): Source dicts built from source_nodes — mutated in place
        """
        if not sources:
            return

        try:
            query_embedding = self.provider.get_embeddings(question)
            collection = self._get_collection()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=self.similarity_top_k,
                include=["distances", "metadatas"]
            )

            # Build a lookup from file_name.. best distance (lowest = most similar)
            # A file can appear in multiple chunks so keep the minimum distance.
            distance_by_file = {}
            for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
                fname = meta.get("file_name", "")
                if fname not in distance_by_file or dist < distance_by_file[fname]:
                    distance_by_file[fname] = dist

            for source in sources:
                fname = source.get("file", "")
                if fname in distance_by_file:
                    raw = 1 / (1 + distance_by_file[fname])
                    source["score"] = round(max(0.0, min(1.0, raw)), 4)

            self.logger.info("Scores enriched from ChromaDB distances")

        except Exception as e:
            self.logger.warning(f"Score enrichment from ChromaDB failed — using 0.0 fallback: {e}")

    def _format_sources_for_provider(self, source_nodes: list) -> tuple:
        """
        Build both the display sources list and the provider sources list from
        retrieved source nodes.

        The display list is used by app.py to render citations. The provider
        list is the format expected by providers with uses_source_list = True
        (e.g. PleiasProvider), where each entry is a plain dict with text and
        metadata fields matching the rag_library contract.

        Args:
            source_nodes (list): NodeWithScore objects from a retriever

        Returns:
            tuple: (display_sources, provider_sources)
                display_sources — list of dicts for app.py
                provider_sources — list of {"text": ..., "metadata": {"source": ...}}
        """
        display_sources = []
        provider_sources = []
        for node in source_nodes:
            display_sources.append({
                "file": node.node.metadata.get("file_name", "Unknown"),
                "page": node.node.metadata.get("page_label", "Unknown"),
                "score": round(node.score, 4) if node.score is not None else 0.0,
                "preview": node.node.text[:150],
                "text": node.node.text[:1000],
            })
            provider_sources.append({
                "text": node.node.text,
                "metadata": {"source": node.node.metadata.get("file_name", "")},
            })
        return display_sources, provider_sources

    def query(self, question: str, chat_history: list = None) -> dict:
        """
        Run a question through the RAG pipeline and return a complete response.

        When the active provider sets uses_source_list = True (e.g. PleiasProvider),
        retrieval is done directly via the retriever and the formatted source list
        is passed to the provider. Otherwise the standard LlamaIndex synthesis
        path is used.

        Args:
            question (str): The user's question
            chat_history (list): List of dicts with 'role' and 'content' keys

        Returns:
            dict: answer, sources, query_time
        """
        self.logger.info(f"Query received: '{question}'")

        history_context = self._build_history_context(
            [] if getattr(self.provider, 'uses_source_list', False) else chat_history
        )
        contextual_question = self._build_contextual_question(question, history_context)

        start = time.time()

        if getattr(self.provider, 'uses_source_list', False):
            source_nodes = self.query_engine.retriever.retrieve(contextual_question)
            sources, provider_sources = self._format_sources_for_provider(source_nodes)
            answer = self.provider.generate(question, sources=provider_sources)
        else:
            response = self.query_engine.query(contextual_question)
            sources = []
            for node in response.source_nodes:
                sources.append({
                    "file": node.metadata.get("file_name", "Unknown"),
                    "page": node.metadata.get("page_label", "Unknown"),
                    "score": round(node.score, 4) if node.score is not None else 0.0,
                    "preview": node.text[:150],
                    "text": node.text[:1000],
                })
            answer = str(response)

        self.last_query_time = round(time.time() - start, 2)
        self._enrich_scores_from_chroma(question, sources)
        self.last_sources = sources
        self.logger.info(f"Query completed | Time: {self.last_query_time}s | Sources retrieved: {len(sources)}")

        return {
            "answer": answer,
            "sources": sources,
            "query_time": self.last_query_time,
        }

    def stream_query(self, question: str, chat_history: list = None):
        """
        Run a question through the RAG pipeline and stream the response token by token.

        Retrieval happens upfront — source nodes are available immediately after the
        streaming response object is created, before any tokens are generated. Sources
        are stored on self.last_sources so app.py can access them without a second query.

        When the active provider sets uses_source_list = True (e.g. PleiasProvider),
        retrieval is done via the retriever directly and sources are passed to
        provider.stream(). The generator contract is identical to the standard path
        so app.py requires no changes.

        Args:
            question (str): The user's question
            chat_history (list): List of dicts with 'role' and 'content' keys

        Yields:
            str: Each token as it is generated
        """
        self.logger.info(f"Stream query received: '{question}'")

        history_context = self._build_history_context(
            [] if getattr(self.provider, 'uses_source_list', False) else chat_history
        )
        contextual_question = self._build_contextual_question(question, history_context)

        start = time.time()

        if getattr(self.provider, 'uses_source_list', False):
            source_nodes = self.streaming_query_engine.retriever.retrieve(contextual_question)
            self.last_sources, provider_sources = self._format_sources_for_provider(source_nodes)
            self._enrich_scores_from_chroma(question, self.last_sources)
            self.logger.info(f"Sources retrieved: {len(self.last_sources)} | Starting token stream")
            for token in self.provider.stream(question, sources=provider_sources):
                yield token
        else:
            # Retrieval happens here synchronously, source_nodes are populated
            # before any tokens are generated
            streaming_response = self.streaming_query_engine.query(contextual_question)

            # Extract sources immediately after retrieval before streaming begins
            self.last_sources = []
            for node in streaming_response.source_nodes:
                self.last_sources.append({
                    "file": node.metadata.get("file_name", "Unknown"),
                    "page": node.metadata.get("page_label", "Unknown"),
                    "score": round(node.score, 4) if node.score is not None else 0.0,
                    "preview": node.text[:150],
                    "text": node.text[:1000],
                })

            self._enrich_scores_from_chroma(question, self.last_sources)
            self.logger.info(f"Sources retrieved: {len(self.last_sources)} | Starting token stream")

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
            "docs_count": self._get_chunk_count(),
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

        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        self.logger.info(f"Loaded {len(docs)} pages from {file_path}")

        collection = self._get_collection()
        chunks_before = collection.count()

        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)

        for doc in docs:
            index.insert(doc)

        new_chunks = collection.count() - chunks_before

        self._build_engines_from_index(index)

        self.logger.info(f"Document added | New chunks: {new_chunks} | Total chunks: {collection.count()}")
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

        collection = self._get_collection()
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

        _, index = self._build_vector_store_and_index()
        self._build_engines_from_index(index)

        return True

    def switch_models(self, llm_model: str, embed_model: str):
        """
        Switch models without restarting the app.

        Updates config values and reloads the provider so the new
        models are used for all subsequent queries.

        Args:
            llm_model (str): New model name
            embed_model (str): New embedding model name
        """
        self.logger.info(f"Switching models | LLM: {llm_model} | Embed: {embed_model}")

        self.llm_model = llm_model
        self.embed_model_name = embed_model

        config = load_config()
        config["llm_model"] = llm_model
        config["embed_model"] = embed_model

        self.provider = get_provider(config)
        Settings.llm = self.provider.llm if hasattr(self.provider, 'llm') else Settings.llm
        Settings.embed_model = self.provider.embed_model

        self.query_engine = self._build_query_engine()
        self.logger.info(f"Model switch complete | LLM: {llm_model} | Embed: {embed_model}")

