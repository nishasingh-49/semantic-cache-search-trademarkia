from sklearn.decomposition import PCA
from data.dataset_loader import load_dataset
from preprocessing.clean_text import clean_text
from embeddings.embedder import Embedder
from clustering.fuzzy_cluster import FuzzyCluster
import numpy as np
def main():
    print("Loading dataset")
    docs, labels, names = load_dataset()
    print("Cleaning documents")
    docs = [clean_text(doc) for doc in docs]
    print("Generating embeddings")
    embedder = Embedder()
    embeddings = embedder.embed_documents(docs)
    print("Reducing embedding dimensions with PCA...")
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)
    print("Running fuzzy clustering...")
    cluster_model = FuzzyCluster(n_clusters=5)
    membership = cluster_model.fit(reduced_embeddings)
    # Find a strongly clustered document
    dominance_scores = []
    for i in range(len(docs)):
        probs = membership[:, i]
        dominance_scores.append((i, np.max(probs)))
    best_doc = sorted(dominance_scores, key=lambda x: x[1], reverse=True)[0][0]
    print("\nExample membership distribution for a strongly clustered document:\n")
    for i, prob in enumerate(membership[:, best_doc]):
        print(f"Cluster {i}: {prob:.3f}")
    print("\nDominant cluster:", np.argmax(membership[:, best_doc]))
    # Show example documents per cluster
    print("\n\nExample documents from each cluster:\n")
    for cluster_id in range(5):
        print(f"\nCluster {cluster_id} examples:\n")
        count = 0
        for doc_index in range(len(docs)):
            dominant = np.argmax(membership[:, doc_index])
            if dominant == cluster_id:
                print("-", docs[doc_index][:200], "\n")
                count += 1
                if count == 3:
                    break
    # Detect boundary documents
    print("\n\nFinding boundary (uncertain) documents...\n")
    uncertainty_scores = []
    for i in range(len(docs)):
        probs = membership[:, i]
        top_two = np.sort(probs)[-2:]
        uncertainty = top_two[1] - top_two[0]
        uncertainty_scores.append((i, uncertainty))
    boundary_docs = sorted(uncertainty_scores, key=lambda x: x[1])[:5]
    print("Documents near cluster boundaries:\n")
    for idx, score in boundary_docs:
        print("Document index:", idx)
        print("Snippet:", docs[idx][:200])
        print("Uncertainty score:", score)
        print()
if __name__ == "__main__":
    main()