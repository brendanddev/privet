
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


class RAGDebugger:
    """
    A debugging and inspection utility for RAG pipelines.
    
    This class provides tools to inspect the contents of a ChromaDB vector store,
    analyze chunk quality, trace query retrieval, and understand how embeddings
    relate to one another.
    
    Attributes:
        client (chromadb.PersistentClient): Connection to the ChromaDB instance
        collection (chromadb.Collection): The document collection being inspected
        query_log (list): A running log of all traced queries during this session
    """

    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "documents"):
        """
        Initialize the debugger and connect to a ChromaDB collection.

        Args:
            chroma_path (str): Path to the persistent ChromaDB directory. Defaults to ./chroma_db
            collection_name (str): Name of the collection to inspect. Defaults to 'documents'
        """
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.query_log = []  # Stores traced queries for the current session

    def summary(self):
        """
        Print a high-level overview of the collection.
        
        Shows total chunk count, indexed documents, chunk size stats,
        and the dimensionality of the stored embeddings.
        """
        total = self.collection.count()

        # Fetch all documents, embeddings, and metadata from the collection
        results = self.collection.get(include=["documents", "embeddings", "metadatas"])

        # Calculate character length of each chunk for size analysis
        chunk_lengths = [len(doc) for doc in results["documents"]]

        # Extract unique filenames from metadata
        files = set(m.get("file_name", "unknown") for m in results["metadatas"])

        print("=" * 50)
        print("RAG DEBUGGER — Collection Summary")
        print("=" * 50)
        print(f"Total chunks:        {total}")
        print(f"Documents indexed:   {', '.join(files)}")
        print(f"Avg chunk length:    {int(np.mean(chunk_lengths))} chars")
        print(f"Min chunk length:    {min(chunk_lengths)} chars")
        print(f"Max chunk length:    {max(chunk_lengths)} chars")
        print(f"Embedding dims:      {len(results['embeddings'][0])}")
        print("=" * 50)

    def inspect_chunks(self, n: int = 5):
        """
        Print a human-readable preview of the first N chunks in the collection.
        
        Useful for verifying that documents were ingested and chunked correctly,
        and for checking what metadata is being stored alongside each chunk.

        Args:
            n (int): Number of chunks to inspect. Defaults to 5.
        """
        results = self.collection.peek(n)

        for i, (doc, embedding, metadata) in enumerate(zip(
            results["documents"],
            results["embeddings"],
            results["metadatas"]
        )):
            print(f"\n--- Chunk {i+1} ---")
            print(f"File:       {metadata.get('file_name')} | Page: {metadata.get('page_label')}")
            print(f"Text:       {doc[:300]}")
            # Only show the first 2 values of the embedding for readability
            print(f"Embedding:  [{embedding[0]:.4f}, {embedding[1]:.4f}, ... ] ({len(embedding)} dims)")

    def chunk_distribution(self):
        """
        Print a simple bar chart showing the distribution of chunk sizes.
        
        Helps identify chunking quality issues:
        - Too many small chunks (< 200 chars) may indicate headers or noise
        - Very large chunks (1000+ chars) may reduce retrieval precision
        """
        results = self.collection.get(include=["documents"])

        # Measure each chunk by character count
        lengths = [len(doc) for doc in results["documents"]]

        # Sort chunks into size buckets
        buckets = {"0-200": 0, "200-500": 0, "500-1000": 0, "1000+": 0}
        for l in lengths:
            if l < 200:
                buckets["0-200"] += 1
            elif l < 500:
                buckets["200-500"] += 1
            elif l < 1000:
                buckets["500-1000"] += 1
            else:
                buckets["1000+"] += 1

        print("\nChunk Size Distribution:")
        for bucket, count in buckets.items():
            bar = "█" * count
            print(f"  {bucket:10} | {bar} ({count})")

    def trace_query(self, query_text: str, n_results: int = 3):
        """
        Embed a query and show which chunks would be retrieved, along with similarity scores.
        
        This simulates exactly what happens during a real RAG query — the query is embedded
        using the same model as the documents, then ChromaDB finds the closest matching chunks
        by vector similarity. The score shows how confident the retrieval is.

        Args:
            query_text (str): The question or query to trace
            n_results (int): Number of top chunks to retrieve. Defaults to 3.
        """
        from llama_index.embeddings.ollama import OllamaEmbedding

        # Use the same embedding model as the one used during ingestion
        embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        # Convert the query text into a vector embedding
        query_embedding = embed_model.get_text_embedding(query_text)

        # Query ChromaDB for the most similar chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\nQuery: '{query_text}'")
        print(f"Top {n_results} retrieved chunks:\n")

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            # ChromaDB returns L2 distance — convert to a 0-1 similarity score
            similarity = 1 - dist
            print(f"  [{i+1}] Score: {similarity:.4f} | File: {meta.get('file_name')} | Page: {meta.get('page_label')}")
            print(f"       {doc[:200]}\n")

        # Log the query for later review
        self.query_log.append({
            "query": query_text,
            "timestamp": datetime.now().isoformat(),
            "results": len(results["documents"][0])
        })

    def chunk_similarity_matrix(self, n: int = 10):
        """
        Print a cosine similarity matrix for the first N chunks.
        
        Cosine similarity measures how semantically similar two chunks are,
        on a scale from 0 (completely different) to 1 (identical meaning).
        High similarity between different chunks may indicate redundant content.

        Args:
            n (int): Number of chunks to compare. Defaults to 10.
        """
        results = self.collection.get(include=["embeddings", "documents"])

        # Take the first N embeddings and convert to a numpy array for computation
        embeddings = np.array(results["embeddings"][:n])

        # Compute pairwise cosine similarity across all selected chunks
        matrix = cosine_similarity(embeddings)

        print(f"\nCosine Similarity Matrix (first {n} chunks):")
        print("     " + "  ".join([f"C{i+1:02}" for i in range(n)]))

        for i, row in enumerate(matrix):
            scores = "  ".join([f"{v:.2f}" for v in row])
            print(f"C{i+1:02}  {scores}")

    def query_history(self):
        """
        Print a log of all queries traced during the current session.
        
        Useful for reviewing what was tested and how many chunks each query retrieved.
        Note: this log resets each time the debugger is instantiated.
        """
        if not self.query_log:
            print("No queries logged yet.")
            return

        print("\nQuery Log:")
        for entry in self.query_log:
            print(f"  [{entry['timestamp']}] '{entry['query']}' — {entry['results']} chunks retrieved")


if __name__ == "__main__":
    debugger = RAGDebugger()

    # Overview of what's stored in the collection
    debugger.summary()

    # Preview the first 3 chunks
    debugger.inspect_chunks(n=3)

    # Visualize chunk size distribution
    debugger.chunk_distribution()

    # Trace a sample query to see which chunks get retrieved and their scores
    debugger.trace_query("What is this project about?")

    # Compare semantic similarity between the first 5 chunks
    debugger.chunk_similarity_matrix(n=5)

    # Print the query log for this session
    debugger.query_history()