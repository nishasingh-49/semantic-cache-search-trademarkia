import numpy as np
import skfuzzy as fuzz
class FuzzyCluster:
    def __init__(self, n_clusters=5):
        """
        Fuzzy C-Means clustering.
        Each document can belong to multiple clusters with
        different membership probabilities instead of a hard label.
        """
        self.n_clusters = n_clusters
        self.centers = None
        self.membership = None
    def fit(self, embeddings):
        data = np.array(embeddings).T
        centers, membership, _, _, _, _, _ = fuzz.cluster.cmeans(
            data,
            c=self.n_clusters,
            m=2,
            error=0.005,
            maxiter=1000
        )
        self.centers = centers
        self.membership = membership
        return membership
    def dominant_cluster(self, doc_index):
        return np.argmax(self.membership[:, doc_index])