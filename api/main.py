from fastapi import FastAPI
from pydantic import BaseModel
from embeddings.embedder import Embedder
from cache.semantic_cache import SemanticCache
from vectordb.faiss_store import VectorDB
from clustering.fuzzy_cluster import FuzzyCluster
import numpy as np
app = FastAPI(title="Semantic Cache Search API")
# Initialize components
embedder = Embedder()
cache = SemanticCache(similarity_threshold=0.7)
vector_db = None
cluster_model = None
class QueryRequest(BaseModel):
    query: str
@app.on_event("startup")
def startup_event():
    global vector_db, cluster_model
    print("Initializing system")
    # Dummy initialization
    vector_db = VectorDB(dim=384)
    cluster_model = FuzzyCluster(n_clusters=5)
    print("System ready")
@app.post("/query")
def query_endpoint(request: QueryRequest):
    query = request.query
    query_embedding = embedder.embed_query(query)
    cached_entry, score = cache.lookup(query_embedding)
    if cached_entry:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached_entry["query"],
            "similarity_score": float(score),
            "result": cached_entry["result"],
            "dominant_cluster": cached_entry["cluster"]
        }
    # Simulate search result (replace with vector DB search if needed)
    result = "semantic search result placeholder"
    # Assign dummy cluster
    dominant_cluster = 0
    cache.add(query, query_embedding, result, dominant_cluster)
    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0,
        "result": result,
        "dominant_cluster": dominant_cluster
    }
@app.get("/cache/stats")
def cache_stats():
    return cache.stats()
@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}