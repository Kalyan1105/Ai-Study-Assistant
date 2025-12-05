import os
import tempfile
from functools import lru_cache
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
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

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache
def get_embedder() -> SentenceTransformer:
    # Lazy-load model on first request to save memory at startup
    return SentenceTransformer(EMBED_MODEL_NAME)


CHROMA_DIR = "./chroma_study_db"
os.makedirs(CHROMA_DIR, exist_ok=True)
client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client_chroma.get_or_create_collection(name="study_notes")


# -------------------------
# Core logic
# -------------------------

def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are a helpful AI study assistant for a CSE student.
Answer ONLY using the context below.
If answer not found, say exactly: "I don't know based on these notes."

Context:
{context}

Question: {question}
"""

    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Study assistant"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        text = p.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += max(chunk_size - overlap, 1)
    return chunks


def build_doc_chunks(files: List[UploadFile]) -> List[Dict]:
    docs = []
    doc_id = 0
    for uploaded in files:
        if not uploaded.filename.lower().endswith(".pdf"):
            continue

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
    return docs


def reset_collection():
    try:
        client_chroma.delete_collection("study_notes")
    except Exception:
        pass
    global collection
    collection = client_chroma.get_or_create_collection(name="study_notes")


def add_documents_to_chroma(docs: List[Dict]):
    if not docs:
        return
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metas = [d["metadata"] for d in docs]

    embeddings = get_embedder().encode(texts, convert_to_numpy=True).tolist()
    collection.add(documents=texts, ids=ids, metadatas=metas, embeddings=embeddings)


def query_study_notes(question: str, k: int = 5):
    q_emb = get_embedder().encode([question], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)

    if not results["documents"]:
        return "I don't know based on these notes.", "", []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    context_parts = []
    chunks = []
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
    k: int = 5


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/build_index")
async def build_index(files: List[UploadFile] = File(...)):
    reset_collection()
    docs = build_doc_chunks(files)
    add_documents_to_chroma(docs)
    return {"message": "Index built", "chunks": len(docs)}


@app.post("/ask")
async def ask(req: AskRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on the server."}
    answer, context, chunks = query_study_notes(req.question, req.k)
    return {"question": req.question, "answer": answer, "context": context, "chunks": chunks}
