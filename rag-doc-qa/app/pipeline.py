"""
pipeline.py — Orchestrator that ties retrieval and generation together.

Calls the retriever to find relevant chunks, then passes them to the
generator to produce a final answer.
"""

from app.retriever import retrieve
from app.generator import generate_answer


def ask(question: str) -> dict:
    """
    Full RAG pipeline: retrieve context → generate answer.

    Parameters
    ----------
    question : str
        The user's natural-language question.

    Returns
    -------
    dict
        {
            "answer": str,       # LLM-generated answer
            "sources": list[dict] # retrieved chunks with text, page, filename, score
        }
    """
    # Step 1 — Retrieve relevant chunks from Endee
    sources = retrieve(question)

    # Step 2 — Generate an answer using Gemini
    answer = generate_answer(question, sources)

    return {
        "answer": answer,
        "sources": sources,
    }
