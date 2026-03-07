from sentence_transformers import SentenceTransformer
class Embedder:
    def __init__(self):
        """
        all-MiniLM-L6-v2 was chosen because:
        - strong semantic similarity performance
        - small (384 dimension vectors)
        - fast enough for real-time APIs
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_documents(self, documents):
        return self.model.encode(documents, show_progress_bar=True)
    def embed_query(self, query):
        return self.model.encode([query])[0]