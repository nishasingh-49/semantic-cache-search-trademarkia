import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class SemanticCache:
    def __init__(self, similarity_threshold=0.70):
        # threshold that decides whether queries are "similar enough"
        self.similarity_threshold = similarity_threshold
        # store cached entries
        self.entries = []
        # stats
        self.hit_count = 0
        self.miss_count = 0
    def lookup(self, query_embedding):
        if len(self.entries) == 0:
            self.miss_count += 1
            return None, None
        embeddings = [entry["embedding"] for entry in self.entries]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        if best_score >= self.similarity_threshold:
            self.hit_count += 1
            return self.entries[best_idx], best_score
        else:
            self.miss_count += 1
            return None, best_score
    def add(self, query, embedding, result, cluster_id):
        self.entries.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster_id
        })
    def stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total else 0
        return {
            "total_entries": len(self.entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }
    def clear(self):
        self.entries = []
        self.hit_count = 0
        self.miss_count = 0