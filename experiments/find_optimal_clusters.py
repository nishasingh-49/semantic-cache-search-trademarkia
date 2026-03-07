from data.dataset_loader import load_dataset
from preprocessing.clean_text import clean_text
from embeddings.embedder import Embedder
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
def main():
    print("Loading dataset")
    docs, labels, names = load_dataset()
    print("Cleaning documents")
    docs = [clean_text(d) for d in docs]
    print("Generating embeddings")
    embedder = Embedder()
    embeddings = embedder.embed_documents(docs)
    data = np.array(embeddings).T
    cluster_range = range(5, 21)
    fpc_scores = []
    print("Testing cluster counts")
    for c in cluster_range:
        _, _, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data,
            c=c,
            m=2,
            error=0.005,
            maxiter=1000
        )
        print(f"Clusters: {c}, FPC: {fpc}")
        fpc_scores.append(fpc)
    # Plot
    plt.figure()
    plt.plot(cluster_range, fpc_scores, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Fuzzy Partition Coefficient (FPC)")
    plt.title("Cluster Quality vs Number of Clusters")
    plt.show()
if __name__ == "__main__":
    main()