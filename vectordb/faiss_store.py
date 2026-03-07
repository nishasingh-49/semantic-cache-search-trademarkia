import faiss
import numpy as np
class VectorDB:
    def __init__(self, dim):
        """
        FAISS is used for efficient similarity search.
        IndexFlatL2 performs exact nearest neighbour search
        which is suitable for datasets of this size (~20k docs).
        """
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
    def add_documents(self, embeddings, documents):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.documents.extend(documents)
    def search(self, query_embedding, k=5):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        results = [self.documents[i] for i in indices[0]]
        return results