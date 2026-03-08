# Semantic Cache Search System

A lightweight semantic search system that combines vector embeddings, fuzzy clustering, and a semantic cache to efficiently serve natural language queries.

The system is built on the **20 Newsgroups dataset** and demonstrates how semantic similarity can be leveraged to reduce redundant computation in search systems.

Instead of relying on exact string matches, the cache identifies **semantically similar queries**, enabling reuse of previously computed results even when users phrase their questions differently.

---

# Problem Statement

Traditional caching mechanisms fail in natural language systems.

Example:

```
User A: "machine learning algorithms"
User B: "algorithms used in machine learning"
```

Although these queries mean the same thing, traditional caches treat them as **different inputs**, forcing redundant computation.

This project solves that problem by introducing a **semantic cache**, where queries are compared using vector embeddings rather than raw text.

---

# Key Features

- Semantic embeddings for documents and queries
- Fuzzy clustering to capture overlapping topic structure
- Custom semantic cache built from scratch
- Efficient vector search using FAISS
- FastAPI-based REST API
- Dockerized deployment

---

# System Architecture

```
User Query
   │
   ▼
Embedding Model
(Sentence Transformers)
   │
   ▼
Semantic Cache Lookup
   │
 ┌─┴─────────────┐
 │               │
Cache Hit     Cache Miss
 │               │
Return         Vector Search
Cached Result     │
                  ▼
           FAISS Vector Index
                  │
                  ▼
           Result Cached
```

---

# Dataset

This project uses the **20 Newsgroups dataset**.

Characteristics:

- ~20,000 documents
- 20 topic categories
- Contains noisy text such as email headers and signatures

Preprocessing removes noise while preserving semantic meaning.

---

# Embedding Generation

Documents and queries are converted into embeddings using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Features:

- 384-dimensional embeddings
- Lightweight model
- Optimized for semantic similarity

---

# Vector Database

Embeddings are stored using **FAISS** (Facebook AI Similarity Search).

Benefits:

- Efficient nearest-neighbor search
- Optimized for high-dimensional vector data
- Fast semantic retrieval

---

# Fuzzy Clustering

The dataset contains overlapping topics, meaning documents may belong to multiple clusters.

Instead of hard clustering, this project uses **Fuzzy C-Means clustering**.

Each document receives a **membership probability for every cluster**.

Example:

```
Cluster 0 → 0.60
Cluster 1 → 0.30
Cluster 2 → 0.10
```

This better reflects real-world semantic relationships.

---

# Semantic Cache

The semantic cache is implemented **from scratch** without using Redis or external caching systems.

Workflow:

1. Convert query to embedding
2. Compare with cached query embeddings
3. Compute cosine similarity
4. If similarity > threshold → cache hit

Example:

```
Query A: machine learning algorithms
Query B: algorithms used in machine learning
```

Even though the wording differs, the embeddings are similar, so the cached result is reused.

Benefits:

- Avoids redundant computation
- Faster responses
- Efficient scaling

---

# API Endpoints

The system exposes a REST API using **FastAPI**.

---

## POST /query

Submit a natural language query.

Example request:

```json
{
  "query": "machine learning algorithms"
}
```

Example response:

```json
{
  "query": "machine learning algorithms",
  "cache_hit": true,
  "matched_query": "machine learning algorithms",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
```

---

## GET /cache/stats

Returns cache statistics.

Example:

```json
{
  "total_entries": 5,
  "hit_count": 2,
  "miss_count": 3,
  "hit_rate": 0.40
}
```

---

## DELETE /cache

Clears the entire cache and resets statistics.

---

# Project Structure

```
semantic-cache-search
│
├── api
│   └── main.py
│
├── cache
│   └── semantic_cache.py
│
├── clustering
│   └── fuzzy_cluster.py
│
├── embeddings
│   └── embedder.py
│
├── vectordb
│   └── faiss_store.py
│
├── preprocessing
│   └── clean_text.py
│
├── experiments
│   └── cluster_analysis.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Running the Project

## Docker (Recommended)

Build the Docker image:

```bash
docker build -t semantic-cache .
```

Run the container:

```bash
docker run -p 8000:8000 semantic-cache
```

Open the API documentation:

```
http://localhost:8000/docs
```

---

## Running Locally

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn api.main:app --reload
```

Open:

```
http://localhost:8000/docs
```

---

# Example Demo

1. Run a query

```
machine learning algorithms
```

Response:

```
cache_hit: false
```

2. Run the same query again

Response:

```
cache_hit: true
```

This demonstrates the semantic cache functionality.

---

# Design Decisions

Embedding Model  
MiniLM provides strong semantic representation while remaining lightweight.

Vector Database  
FAISS enables fast similarity search across large embedding collections.

Clustering  
Fuzzy clustering captures overlapping semantic topics better than hard clustering.

Caching Strategy  
Embedding similarity enables reuse of results across paraphrased queries.

Deployment  
Docker ensures reproducible builds and easy deployment.

---

# Future Improvements

- distributed vector search
- LRU eviction strategy for cache entries
- adaptive similarity thresholds
- cluster-aware cache partitioning
- evaluation metrics for retrieval quality

---

# Author

Nisha Singh
Computer Science Student  
Machine Learning & Systems Enthusiast
