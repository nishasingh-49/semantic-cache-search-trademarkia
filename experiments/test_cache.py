from embeddings.embedder import Embedder
from cache.semantic_cache import SemanticCache
embedder = Embedder()
cache = SemanticCache()
queries = [
    "computer graphics rendering",
    "image rendering algorithms",
    "how to cook pasta"
]
for q in queries:
    embedding = embedder.embed_query(q)
    entry, score = cache.lookup(embedding)
    if entry:
        print("CACHE HIT")
        print("Matched query:", entry["query"])
        print("Similarity:", score)
    else:
        print("CACHE MISS")
        result = "dummy search result"
        cache.add(q, embedding, result, cluster_id=0)