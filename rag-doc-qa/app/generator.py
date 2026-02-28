"""
generator.py — LLM answer generation using Google Gemini.

Builds a RAG prompt from retrieved chunks and sends it to
Gemini 2.0 Flash to produce a grounded answer.
"""

import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ---------------------------------------------------------------------------
# Configure Gemini
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your-gemini-api-key-here":
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add your key to the .env file. "
                "Get a free key at https://aistudio.google.com"
            )
        _client = genai.Client(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, chunks: list[dict]) -> str:
    """Construct the RAG prompt with context chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Chunk {i} | Page {chunk.get('page', '?')} | "
            f"File: {chunk.get('filename', '?')}]\n{chunk.get('text', '')}"
        )
    context_block = "\n\n".join(context_parts)

    prompt = (
        "Answer the following question using ONLY the context provided below. "
        "If the answer cannot be found in the context, clearly state that "
        "the information is not available in the provided document.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Generate an answer to *question* grounded on the retrieved *chunks*.

    Parameters
    ----------
    question : str
        The user's question.
    chunks : list[dict]
        Retrieved context chunks (each with text, page, filename, score).

    Returns
    -------
    str
        The LLM-generated answer.
    """
    client = _get_client()
    prompt = _build_prompt(question, chunks)

    # Retry up to 3 times on rate-limit (429) errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-09-2025",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 20 * (attempt + 1)  # 20s, 40s, 60s
                time.sleep(wait)
            else:
                raise
