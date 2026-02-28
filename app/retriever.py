"""
retriever.py — Semantic search against the Endee vector database.

Embeds the user's question and queries the "documents" index for
the top-k most similar chunks.
"""

from sentence_transformers import SentenceTransformer
from endee import Endee

INDEX_NAME = "documents"
TOP_K = 5

# ---------------------------------------------------------------------------
# Singleton model loader (reuses the same model instance as ingestor)
# ---------------------------------------------------------------------------
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Search the Endee "documents" index for chunks similar to *question*.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    top_k : int
        Number of top results to return (default 5).

    Returns
    -------
    list[dict]
        Each dict has keys: text, page, filename, score.
    """
    # 1. Embed the question
    model = _get_model()
    query_vector = model.encode(question, convert_to_numpy=True).tolist()

    # 2. Query Endee
    client = Endee()  # local dev, no token
    index = client.get_index(name=INDEX_NAME)
    results = index.query(vector=query_vector, top_k=top_k)

    # 3. Format results
    chunks = []
    for item in results:
        meta = item.get("meta", {})
        chunks.append({
            "text": meta.get("text", ""),
            "page": meta.get("page", 0),
            "filename": meta.get("filename", ""),
            "score": item.get("similarity", 0.0),
        })
    return chunks
