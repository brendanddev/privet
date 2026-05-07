class Ingestor:
    def __init__(self, chunker, embedder, store):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def ingest(self, path: str):
        pass