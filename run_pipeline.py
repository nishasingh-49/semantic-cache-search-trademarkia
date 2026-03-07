from data.dataset_loader import load_dataset
from preprocessing.clean_text import clean_text
from embeddings.embedder import Embedder
from vectordb.faiss_store import VectorDB
def main():
    print("Loading dataset")
    docs, labels, names = load_dataset()
    print("Cleaning documents")
    cleaned = [clean_text(doc) for doc in docs]
    print("Generating embeddings")
    embedder = Embedder()
    embeddings = embedder.embed_documents(cleaned)
    print("Building vector database")
    vectordb = VectorDB(dim=len(embeddings[0]))
    vectordb.add_documents(embeddings, cleaned)
    print("Vector database ready")
    query = "computer graphics algorithms"
    query_embedding = embedder.embed_query(query)
    results = vectordb.search(query_embedding)
    print("\nTop results:")
    for r in results[:3]:
        print("-", r[:200])
if __name__ == "__main__":
    main()