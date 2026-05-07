class Retriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str):
        pass