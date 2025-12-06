import os
import tempfile
from functools import lru_cache
from typing import List, Dict, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PyPDF2 import PdfReader
from PyPDF2.errors import DependencyError as PdfDependencyError
import chromadb
from fastembed import TextEmbedding
from groq import Groq, APIStatusError
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

# fastembed will download this ONNX model once, then cache it
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
collection = client_chroma.get_or_create_collection(name="study_notes")

# IMPORTANT:
# This flag says "has someone built an index in THIS server process?"
# It starts False every time the app starts, so old persisted data is
# not considered "ready" until /build_index is called.
INDEX_READY: bool = False


# -------------------------
# PDF utilities
# -------------------------

def read_pdf(path: str) -> str:
    """
    Read a PDF file into a single text string.
    - Gracefully skips encrypted / unsupported PDFs (e.g., AES without PyCryptodome).
    """
    try:
        reader = PdfReader(path)
    except Exception as e:
        print(f"[PDF] Error opening {path}: {e}")
        return ""

    # Handle encryption (if any)
    if getattr(reader, "is_encrypted", False):
        try:
            # Try empty password (many academic PDFs have blank protection)
            reader.decrypt("")
        except Exception as e:
            print(f"[PDF] Skipping encrypted PDF (can't decrypt): {e}")
            return ""

    pages: List[str] = []
    try:
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
    except PdfDependencyError as e:
        # This happens for AES-encrypted PDFs when PyCryptodome is missing
        print(f"[PDF] Skipping PDF due to encryption dependency: {e}")
        return ""
    except Exception as e:
        print(f"[PDF] Error while reading pages: {e}")
        return ""

    return "\n".join(pages).strip()


def chunk_text(
    text: str,
    max_chars: int = 900,
    overlap_chars: int = 150,
) -> List[str]:
    """
    Character-based chunking:
    - Smaller chunks → sharper embeddings, fewer tokens per chunk.
    - Overlap keeps context continuity.
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n")
    n = len(text)
    chunks: List[str] = []

    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        # Move back a bit for overlap
        start = max(0, end - overlap_chars)

    return chunks


def build_doc_chunks(files: List[UploadFile]) -> List[Dict]:
    """
    Turn uploaded PDFs into a list of chunk dicts:
    {
        "id": "doc0_chunk3",
        "text": "...",
        "metadata": {
            "source_file": "os_notes.pdf",
            "chunk_index": 3,
        },
    }
    """
    docs: List[Dict] = []
    doc_id = 0

    for uploaded in files:
        filename = uploaded.filename or "uploaded.pdf"

        if not filename.lower().endswith(".pdf"):
            print(f"[UPLOAD] Skipping non-PDF file: {filename}")
            continue

        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.file.read())
            tmp_path = tmp.name

        full_text = read_pdf(tmp_path)
        if not full_text:
            print(f"[UPLOAD] No text extracted from: {filename}")
            continue

        chunks = chunk_text(full_text)

        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"doc{doc_id}_chunk{idx}",
                    "text": chunk,
                    "metadata": {
                        "source_file": filename,
                        "chunk_index": idx,
                    },
                }
            )
        doc_id += 1

    print(f"[INDEX] Built {len(docs)} chunks from {doc_id} PDFs")
    return docs


# -------------------------
# ChromaDB helpers
# -------------------------

def reset_collection():
    """Drop and recreate the 'study_notes' collection, and mark index as not ready."""
    global collection, INDEX_READY

    try:
        client_chroma.delete_collection("study_notes")
        print("[CHROMA] Existing collection deleted.")
    except Exception:
        # It might not exist yet; ignore.
        pass

    collection = client_chroma.get_or_create_collection(name="study_notes")
    INDEX_READY = False
    print("[CHROMA] Fresh collection created; INDEX_READY = False.")


def add_documents_to_chroma(docs: List[Dict]):
    """Embed and add documents into Chroma collection."""
    global INDEX_READY

    if not docs:
        print("[CHROMA] No docs to add.")
        INDEX_READY = False
        return

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metas = [d["metadata"] for d in docs]

    embedder = get_embedder()
    # fastembed.embed() → generator of numpy arrays
    embeddings = [vec.tolist() for vec in embedder.embed(texts)]

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metas,
        embeddings=embeddings,
    )

    INDEX_READY = True
    print(f"[CHROMA] Added {len(docs)} chunks to collection; INDEX_READY = True.")


# -------------------------
# LLM / Answer generation
# -------------------------

def _build_limited_context(
    docs: List[str],
    metas: List[Dict],
    max_context_chars: int = 12000,
) -> Tuple[str, List[Dict]]:
    """
    Build a nicely annotated context string, but hard-limit total characters
    so we don't blow Groq's token limits.
    """
    context_parts: List[str] = []
    used_chunks: List[Dict] = []
    total_chars = 0

    for i, d in enumerate(docs):
        meta = metas[i]
        src = meta.get("source_file", "unknown")
        idx = meta.get("chunk_index", -1)

        decorated = f"[Source: {src} | chunk {idx}]\n{d}\n\n"

        if total_chars + len(decorated) > max_context_chars:
            # Stop adding more chunks once we hit the budget
            print(f"[CONTEXT] Reached context char limit at chunk {i}.")
            break

        context_parts.append(decorated)
        used_chunks.append({"text": d, "metadata": meta})
        total_chars += len(decorated)

    context = "".join(context_parts).strip()
    return context, used_chunks


def generate_answer(context: str, question: str) -> str:
    """
    Call Groq LLM on the retrieved context.
    - Includes safety for token limit errors (413).
    """
    if not context.strip():
        return "I don't know based on these notes."

    prompt = f"""
You are a helpful AI study assistant for a CSE student.

Use ONLY the context below to answer the question.
If the answer is not clearly present in the context, say exactly:
"I don't know based on these notes."

Be concise, structured, and clear.

Context:
{context}

Question: {question}
"""

    try:
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
    except APIStatusError as e:
        print(f"[GROQ] APIStatusError: {e}")
        # 413 or similar token issue → tell user to narrow question
        return (
            "Sorry, this question plus the retrieved notes are too large "
            "for the current model limits. Please try a more specific question."
        )
    except Exception as e:
        print(f"[GROQ] Unexpected error: {e}")
        return "Sorry, something went wrong while generating the answer."


def query_study_notes(question: str, k: int = 3):
    """
    Retrieve top-k relevant chunks from Chroma and ask Groq.
    We also:
    - Clamp k between 1 and 5 (to avoid huge context).
    - Hard-limit context size in characters to avoid 413 errors.
    """
    max_k = 5
    k = max(1, min(k, max_k))

    embedder = get_embedder()

    # Embed single question
    q_emb = next(embedder.embed([question])).tolist()

    results = collection.query(query_embeddings=[q_emb], n_results=k)

    if not results or not results.get("documents"):
        return "I don't know based on these notes.", "", []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    # Build limited context (hard char budget)
    context, used_chunks = _build_limited_context(
        docs=docs,
        metas=metas,
        max_context_chars=12000,  # ~3000 tokens of context → safe vs 6000 TPM
    )

    answer = generate_answer(context, question)

    # Attach IDs back to used chunks
    chunks_with_ids: List[Dict] = []
    for i, chunk in enumerate(used_chunks):
        if i < len(ids):
            chunk_id = ids[i]
            chunks_with_ids.append(
                {"id": chunk_id, "text": chunk["text"], "metadata": chunk["metadata"]}
            )

    return answer, context, chunks_with_ids


# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(title="AI Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for testing; tighten in prod later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    # Optional: frontend can still send 'k'; we clamp it internally
    k: int = 3


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/build_index")
async def build_index(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDFs and build/update the Chroma index.
    Previous index is cleared each time.
    """
    reset_collection()
    docs = build_doc_chunks(files)
    add_documents_to_chroma(docs)

    if not docs:
        return {
            "message": "No readable text found in the uploaded PDFs. "
                       "Please check your files and try again.",
            "chunks": 0,
        }

    return {"message": "Index built", "chunks": len(docs)}


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Ask a question over the uploaded notes.
    """
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on the server."}

    # 🔒 NEW: if no index has been built in this server process,
    # tell the user to upload PDFs and build the index first.
    global INDEX_READY
    if not INDEX_READY:
        return {
            "error": (
                "No notes indexed yet. "
                "Please upload one or more PDFs and click 'Build / Rebuild Index' first."
            )
        }

    answer, context, chunks = query_study_notes(req.question, k=req.k)

    return {
        "question": req.question,
        "answer": answer,
        "context": context,
        "chunks": chunks,
    }
