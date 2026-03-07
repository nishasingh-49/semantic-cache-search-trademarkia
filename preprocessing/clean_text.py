import re
def clean_text(text):
    """
    Basic preprocessing.
    Lowercasing ensures consistent embeddings.
    Removing punctuation reduces noise.
    """
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()