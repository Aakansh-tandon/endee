"""
ingestor.py — PDF ingestion pipeline.

Reads a PDF, splits text into overlapping chunks, generates embeddings,
and upserts vectors into the Endee vector database.
"""

import uuid
import fitz  # PyMuPDF
import requests
from sentence_transformers import SentenceTransformer
from endee import Endee

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDEX_NAME = "documents"
EMBEDDING_DIM = 384
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlapping characters between consecutive chunks
BATCH_SIZE = 500       # max vectors per upsert call (Endee limit is 1000)
ENDEE_BASE_URL = "http://localhost:8080/api/v1"

# ---------------------------------------------------------------------------
# Singleton-style model loader (cached across calls in the same process)
# ---------------------------------------------------------------------------
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Return a list of {'page': int, 'text': str} for every page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_text(pages: list[dict], chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split page texts into overlapping character-level chunks.
    Each chunk carries the originating page number.
    """
    chunks = []
    for page_info in pages:
        text = page_info["text"]
        page = page_info["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({"text": chunk, "page": page})
            start += chunk_size - overlap
    return chunks


def _ensure_index(client: Endee) -> None:
    """Create the Endee index if it does not already exist."""
    # Bypass SDK validation — create index via direct HTTP call
    # because the SDK expects "int8" but the server expects "int8d"
    resp = requests.post(
        f"{ENDEE_BASE_URL}/index/create",
        headers={"Content-Type": "application/json"},
        json={
            "index_name": INDEX_NAME,
            "dim": EMBEDDING_DIM,
            "space_type": "cosine",
            "precision": "int8d",
            "M": 16,
            "ef_con": 128,
        },
    )
    # 409 = index already exists, which is fine
    if resp.status_code == 409:
        return
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Main ingestion entry-point
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str, filename: str) -> int:
    """
    End-to-end ingestion of a single PDF file.

    Parameters
    ----------
    pdf_path : str
        Local filesystem path to the PDF.
    filename : str
        Original filename (stored as metadata).

    Returns
    -------
    int
        Number of chunks ingested.
    """
    # 1. Extract text
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return 0

    # 2. Chunk
    chunks = chunk_text(pages)
    if not chunks:
        return 0

    # 3. Embed
    model = _get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # 4. Prepare vectors for Endee
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": str(uuid.uuid4()),
            "vector": embeddings[i].tolist(),
            "meta": {
                "text": chunk["text"],
                "page": chunk["page"],
                "filename": filename,
            },
        })

    # 5. Upsert into Endee (in batches)
    client = Endee()  # no token for local dev
    _ensure_index(client)
    index = client.get_index(name=INDEX_NAME)

    for start in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[start : start + BATCH_SIZE]
        index.upsert(batch)

    return len(vectors)
