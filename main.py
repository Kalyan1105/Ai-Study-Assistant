import os
import tempfile
from functools import lru_cache
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError, DependencyError

import chromadb
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv


# -------------------------
# Config & Setup
# -------------------------

load_dotenv()  # for local dev (.env); Render uses real env vars

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Don't crash on import in Render logs, but be explicit
    print("WARNING: GROQ_API_KEY not set!")

groq_client = Groq(api_key=GROQ_API_KEY)

# You can change this to another supported fastembed model if you want later.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache
def get_embedder() -> TextEmbedding:
    """
    Lazy-load the embedding model on first use.
    fastembed uses ONNX CPU models (no torch/CUDA) → light & fast for Render.
    """
    return TextEmbedding(model_name=EMBED_MODEL_NAME)


CHROMA_DIR = "./chroma_study_db"
os.makedirs(CHROMA_DIR, exist_ok=True)
client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)

# Single shared collection for now (multi-user aware design can come later)
collection = client_chroma.get_or_create_collection(name="study_notes")


# -------------------------
# Core logic
# -------------------------

def generate_answer(context: str, question: str) -> str:
    """
    Call Groq LLM with retrieved context + question.
    """
    prompt = f"""
You are a helpful AI study assistant for a CSE student.
Answer ONLY using the context below.
If the answer is not in the context, say exactly:
"I don't know based on these notes."

Context:
{context}

Question: {question}
"""

    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful study assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def read_pdf(path: str) -> str:
    """
    Read text from a PDF.
    - Supports encrypted PDFs that don't require a user password (AES via pycryptodome).
    - If the PDF is password protected, we raise a user-friendly error.
    """
    try:
        reader = PdfReader(path)
    except PdfReadError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read PDF: {e}",
        )

    # Handle encrypted PDFs
    if reader.is_encrypted:
        # Try with empty password (for some owner-protected PDFs)
        try:
            result = reader.decrypt("")
        except Exception:
            result = 0

        if result == 0:
            # Proper password-protected PDF, we can't read it
            raise HTTPException(
                status_code=400,
                detail="This PDF is password-protected. Please upload an unencrypted file.",
            )

    pages: List[str] = []
    try:
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
    except DependencyError:
        # pycryptodome missing or AES issue – but we already install pycryptodome
        raise HTTPException(
            status_code=500,
            detail="Failed to decrypt this PDF. Please try another file.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while reading the PDF: {e}",
        )

    text = "\n".join(pages).strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this PDF. It may be scanned images only.",
        )

    return text


def chunk_text(text: str) -> List[str]:
    """
    Auto-chunking logic so the user doesn't have to tune anything.

    Strategy:
    - If the text is small, return 1 chunk.
    - Otherwise, use a dynamic chunk size with overlap.
    """
    words = text.split()
    n = len(words)

    if n == 0:
        return []

    # Heuristic: choose chunk size based on document length
    if n <= 800:
        # Small doc → one chunk
        return [" ".join(words)]

    if n <= 4000:
        chunk_size = 500
    else:
        chunk_size = 800

    overlap = int(chunk_size * 0.2)  # 20% overlap
    chunks = []
    start = 0
    while start < n:
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += max(chunk_size - overlap, 1)

    return chunks


def build_doc_chunks(files: List[UploadFile]) -> List[Dict]:
    """
    Accept multiple PDFs, extract text, chunk them, and return list of chunk dicts.
    """
    docs: List[Dict] = []
    doc_id = 0

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded. Please upload at least one PDF.",
        )

    for uploaded in files:
        if not uploaded.filename.lower().endswith(".pdf"):
            # Skip non-PDFs instead of crashing
            continue

        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.file.read())
            tmp_path = tmp.name

        full_text = read_pdf(tmp_path)
        chunks = chunk_text(full_text)

        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"doc{doc_id}_chunk{idx}",
                    "text": chunk,
                    "metadata": {
                        "source_file": uploaded.filename,
                        "chunk_index": idx,
                    },
                }
            )

        doc_id += 1

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No valid text content found in the uploaded PDFs.",
        )

    return docs


def reset_collection():
    """
    Clear the Chroma collection.
    You can call this before re-building everything from scratch.
    """
    try:
        client_chroma.delete_collection("study_notes")
    except Exception:
        # If it doesn't exist yet, ignore
        pass
    global collection
    collection = client_chroma.get_or_create_collection(name="study_notes")


def add_documents_to_chroma(docs: List[Dict]):
    if not docs:
        return

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metas = [d["metadata"] for d in docs]

    embedder = get_embedder()
    # fastembed returns a generator of numpy arrays → convert to list of lists
    embeddings = [vec.tolist() for vec in embedder.embed(texts)]

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metas,
        embeddings=embeddings,
    )


def query_study_notes(question: str):
    """
    Retrieve top chunks and generate an answer.
    We auto-choose k based on how many chunks exist (user doesn't control this).
    """
    embedder = get_embedder()

    total_docs = collection.count()
    if total_docs == 0:
        return (
            "I don't know based on these notes.",
            "",
            [],
        )

    # Auto choose k (number of chunks to retrieve)
    # - At most 5
    # - At least 1
    # - And not more than total_docs
    max_k = 5
    k = min(max_k, max(1, total_docs))

    # Embed single question
    q_emb = next(embedder.embed([question])).tolist()

    results = collection.query(query_embeddings=[q_emb], n_results=k)

    if not results["documents"]:
        return "I don't know based on these notes.", "", []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    context_parts: List[str] = []
    chunks: List[Dict] = []
    for i, d in enumerate(docs):
        meta = metas[i]
        src = meta.get("source_file", "unknown")
        idx = meta.get("chunk_index", -1)
        context_parts.append(f"[Source: {src} | chunk {idx}]\n{d}\n")
        chunks.append({"id": ids[i], "text": d, "metadata": meta})

    context = "\n\n".join(context_parts)
    answer = generate_answer(context, question)
    return answer, context, chunks


# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(title="AI Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for testing; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    # k is kept for backwards compatibility with old frontend,
    # but backend now auto-chooses number of chunks and ignores it.
    k: int | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/build_index")
async def build_index(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDFs and build/update the Chroma index.

    Current behavior:
    - Clears the old index (reset_collection).
    - Indexes ALL uploaded PDFs together.
    """
    reset_collection()
    docs = build_doc_chunks(files)
    add_documents_to_chroma(docs)
    return {"message": "Index built", "chunks": len(docs)}


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Ask a question over the uploaded notes.
    Chunk retrieval count is auto-decided by the backend.
    """
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on the server."}
    answer, context, chunks = query_study_notes(req.question)
    return {
        "question": req.question,
        "answer": answer,
        "context": context,
        "chunks": chunks,
    }
