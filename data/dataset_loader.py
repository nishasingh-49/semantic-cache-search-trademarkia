from sklearn.datasets import fetch_20newsgroups
def load_dataset():
    """
    Loading the 20 newsgroups dataset.
    We remove headers, footers and quotes because they contain
    metadata and reply chains that do not represent the semantic
    content of the document itself.
    """
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )
    documents = dataset.data
    labels = dataset.target
    label_names = dataset.target_names
    return documents, labels, label_names