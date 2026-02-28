"""
streamlit_app.py — Web UI for the RAG Document Q&A system.

Sidebar  : Upload a PDF and ingest it into the vector database.
Main area: Ask a question and get an AI-generated answer with sources.
"""

import sys
import os
import tempfile

# Ensure the project root is on the Python path so `app.*` imports work
# when running via `streamlit run ui/streamlit_app.py` from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.ingestor import ingest_pdf
from app.pipeline import ask

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG Document Q&A")
st.caption("Upload a PDF, then ask natural-language questions about it.")

# ---------------------------------------------------------------------------
# Sidebar — PDF Upload & Ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.success(f"File ready: **{uploaded_file.name}**")

        if st.button("🔄 Ingest Document", use_container_width=True):
            with st.spinner("Extracting text, chunking, embedding & storing..."):
                # Write uploaded bytes to a temp file so PyMuPDF can open it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    num_chunks = ingest_pdf(tmp_path, uploaded_file.name)
                    st.success(
                        f"✅ Ingested **{num_chunks}** chunks from "
                        f"**{uploaded_file.name}** into the vector database."
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                finally:
                    os.unlink(tmp_path)

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Upload a PDF and click *Ingest Document*.\n"
        "2. The text is chunked and stored as vectors in **Endee**.\n"
        "3. Ask a question — the system retrieves relevant chunks and "
        "generates an answer with **Gemini 2.0 Flash**."
    )

# ---------------------------------------------------------------------------
# Main area — Question & Answer
# ---------------------------------------------------------------------------
st.subheader("❓ Ask a Question")

question = st.text_input(
    "Type your question below:",
    placeholder="e.g. What is the main conclusion of the paper?",
)

if st.button("🚀 Get Answer", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask(question)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        # ----- Display answer -----
        st.markdown("### 💡 Answer")
        st.markdown(result["answer"])

        # ----- Display source chunks -----
        st.markdown("---")
        st.markdown("### 📚 Source Chunks")

        sources = result.get("sources", [])
        if not sources:
            st.info("No source chunks were returned.")
        else:
            for i, chunk in enumerate(sources, start=1):
                score_pct = f"{chunk['score']:.4f}"
                label = (
                    f"Chunk {i} — Page {chunk['page']} | "
                    f"File: {chunk['filename']} | "
                    f"Similarity: {score_pct}"
                )
                with st.expander(label):
                    st.write(chunk["text"])
