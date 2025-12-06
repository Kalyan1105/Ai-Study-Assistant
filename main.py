import os
import tempfile
from functools import lru_cache
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PyPDF2 import PdfReader
import chromadb
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv

# -------------------------
# Config & Setup
# -------------------------

load_dotenv()  # local .env; Render uses env vars

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set!")

groq_client = Groq(api_key=GROQ_API_KEY)

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking / retrieval config (medium chunks)
CHUNK_SIZE = 260          # medium chunk (≈ 180–220 tokens)
CHUNK_OVERLAP = 70        # still some continuity
MAX_TOTAL_CHUNKS = 550    # a bit more chunks allowed (still safe)
DEFAULT_TOP_K = 4         # 4 best chunks per question
MAX_CONTEXT_CHARS = 6000  # keep this; protects Groq from 413


@lru_cache
def get_embedder() -> TextEmbedding:
    """
    Lazy-load the embedding model on first use.
    fastembed uses ONNX CPU models (no torch/CUDA) → light & fast for Render.
    """
    print("[EMBED] Loading fastembed model...")
    emb = TextEmbedding(model_name=EMBED_MODEL_NAME)
    print("[EMBED] Model loaded.")
    return emb


# Persistent Chroma DB
CHROMA_DIR = "./chroma_study_db"
os.makedirs(CHROMA_DIR, exist_ok=True)
client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client_chroma.get_or_create_collection(name="study_notes")


# -------------------------
# Core logic
# -------------------------

def generate_answer(context: str, question: str) -> str:
    """
    Call Groq LLM with retrieved context.
    We keep context length limited to avoid 413 errors.
    """
    # Safety: hard truncate context by characters
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are a helpful AI study assistant for a CSE student.

You MUST follow these rules:
- Answer ONLY using the information in the context.
- If the answer is not clearly found in the context, reply exactly:
  "I don't know based on these notes."

Context (from the student's notes):
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
    Read a PDF into plain text.
    Handles simple encrypted PDFs by trying empty password.
    If decryption fails, returns empty string.
    """
    try:
        reader = PdfReader(path)

        # Handle encrypted PDFs (no password)
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # try blank password
            except Exception:
                print(f"[PDF] Encrypted PDF could not be decrypted: {path}")
                return ""

        pages = []
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        print(f"[PDF] Error reading {path}: {e}")
        return ""


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Simple word-based chunking with overlap.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        # next chunk starts a bit before the end (overlap)
        start += max(chunk_size - overlap, 1)

    return chunks


def build_doc_chunks(files: List[UploadFile]) -> List[Dict]:
    """
    Turn uploaded PDFs into a capped list of text chunks.
    Applies MAX_TOTAL_CHUNKS to keep index bounded.
    """
    docs: List[Dict] = []
    doc_id = 0

    for uploaded in files:
        if not uploaded.filename.lower().endswith(".pdf"):
            print(f"[UPLOAD] Skipping non-PDF file: {uploaded.filename}")
            continue

        if len(docs) >= MAX_TOTAL_CHUNKS:
            print("[INDEX] Reached MAX_TOTAL_CHUNKS, ignoring remaining files.")
            break

        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.file.read())
            tmp_path = tmp.name

        full_text = read_pdf(tmp_path)
        if not full_text.strip():
            print(f"[INDEX] No text extracted from {uploaded.filename}, skipping.")
            continue

        chunks = chunk_text(full_text)

        for idx, chunk in enumerate(chunks):
            if len(docs) >= MAX_TOTAL_CHUNKS:
                print("[INDEX] Reached MAX_TOTAL_CHUNKS while chunking, stopping.")
                break

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

    print(f"[INDEX] Prepared {len(docs)} chunks from {doc_id} PDFs")
    return docs


def reset_collection():
    """
    Drop and recreate the global Chroma collection.
    Called every time the user clicks Build / Rebuild Index.
    """
    global collection
    try:
        client_chroma.delete_collection("study_notes")
        print("[CHROMA] Existing collection deleted.")
    except Exception:
        print("[CHROMA] No existing collection to delete (first run).")

    collection = client_chroma.get_or_create_collection(name="study_notes")
    print("[CHROMA] Fresh collection created.")


def add_documents_to_chroma(docs: List[Dict]):
    """
    Embed and add chunks to Chroma.
    Obeys MAX_TOTAL_CHUNKS via build_doc_chunks.
    """
    if not docs:
        print("[CHROMA] No documents to add.")
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

    print(f"[CHROMA] Added {len(texts)} chunks to collection.")


def query_study_notes(question: str, max_k: int = DEFAULT_TOP_K):
    """
    Run a similarity search over the indexed notes and ask the LLM.
    """
    total_chunks = collection.count()
    if total_chunks == 0:
        # handled again at endpoint level, but keep safe here
        msg = (
            "Please upload at least one PDF and build the index "
            "before asking questions."
        )
        return msg, "", []

    k = min(max_k, total_chunks)

    embedder = get_embedder()
    q_emb = next(embedder.embed([question])).tolist()

    results = collection.query(query_embeddings=[q_emb], n_results=k)

    if not results["documents"] or not results["documents"][0]:
        return "I don't know based on these notes.", "", []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    context_parts: List[str] = []
    chunks: List[Dict] = []
    current_len = 0

    for i, d in enumerate(docs):
        meta = metas[i]
        src = meta.get("source_file", "unknown")
        idx = meta.get("chunk_index", -1)

        snippet = f"[Source: {src} | chunk {idx}]\n{d}\n\n"

        # Respect MAX_CONTEXT_CHARS
        if current_len + len(snippet) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - current_len
            if remaining > 0:
                snippet = snippet[:remaining]
                context_parts.append(snippet)
                current_len += len(snippet)
            print("[CONTEXT] Reached MAX_CONTEXT_CHARS while building context.")
            break

        context_parts.append(snippet)
        current_len += len(snippet)

        chunks.append({"id": ids[i], "text": d, "metadata": meta})

    context = "".join(context_parts)
    answer = generate_answer(context, question)
    return answer, context, chunks


# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(title="AI Study Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for testing; tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def warmup():
    """
    Warmup hook:
    - Loads embedder
    - Runs a dummy embedding
    - Touches the Chroma collection
    This removes the "first call is slow" feeling.
    """
    try:
        print("[WARMUP] Starting warmup...")
        emb = get_embedder()
        _ = list(emb.embed(["warmup"]))
        _ = collection.count()
        print("[WARMUP] Warmup complete.")
    except Exception as e:
        print(f"[WARMUP] Warmup failed (non-fatal): {e}")


@app.get("/health")
async def health():
    total_chunks = 0
    try:
        total_chunks = collection.count()
    except Exception:
        pass
    return {"status": "ok", "chunks_indexed": total_chunks}


@app.post("/build_index")
async def build_index(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDFs and build the Chroma index.
    Previous index is cleared each time.
    """
    if not files:
        return {"message": "No files uploaded.", "chunks": 0}

    reset_collection()
    docs = build_doc_chunks(files)
    add_documents_to_chroma(docs)

    return {
        "message": "Index built",
        "chunks": len(docs),
        "max_chunks": MAX_TOTAL_CHUNKS,
    }


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Ask a question over the uploaded notes.
    If no index is built yet, tell the user to upload PDFs first.
    """
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on the server."}

    try:
        total = collection.count()
    except Exception:
        total = 0

    if total == 0:
        msg = (
            "Please upload at least one PDF and click "
            "'Build / Rebuild Index' before asking questions."
        )
        return {
            "question": req.question,
            "answer": msg,
            "context": "",
            "chunks": [],
        }

    answer, context, chunks = query_study_notes(req.question)
    return {
        "question": req.question,
        "answer": answer,
        "context": context,
        "chunks": chunks,
    }
