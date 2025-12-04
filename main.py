import os
import tempfile
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq


# -------------------------
# Config & Setup
# -------------------------

# Read Groq API key from environment (set in Render dashboard
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Chroma persistent directory (Render has ephemeral disk but it's okay for demo)
CHROMA_DIR = "./chroma_study_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client_chroma.get_or_create_collection(name="study_notes")


# -------------------------
# Core Logic
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

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Study assistant"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            texts.append(txt)
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_doc_chunks(files: List[UploadFile]) -> List[Dict]:
    docs = []
    doc_id = 0

    for uploaded_file in files:
        if not uploaded_file.filename.lower().endswith(".pdf"):
            continue

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = uploaded_file.file.read()
            tmp.write(content)
            tmp_path = tmp.name

        full_text = read_pdf(tmp_path)
        chunks = chunk_text(full_text)

        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"doc{doc_id}_chunk{idx}",
                    "text": chunk,
                    "metadata": {
                        "source_file": uploaded_file.filename,
                        "chunk_index": idx,
                    },
                }
            )
        doc_id += 1

    return docs


def reset_collection():
    try:
        client_chroma.delete_collection("study_notes")
    except:
        pass
    global collection
    collection = client_chroma.get_or_create_collection(name="study_notes")


def add_documents_to_chroma(docs: List[Dict]):
    if not docs:
        return

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metas = [d["metadata"] for d in docs]

    embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metas,
        embeddings=embeddings,
    )


def query_study_notes(question: str, k: int = 5):
    q_emb = embedder.encode([question], convert_to_numpy=True).tolist()[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )

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
    answer = generate_answer(context=context, question=question)

    return answer, context, chunks


# -------------------------
# FastAPI App
# -------------------------

app = FastAPI(title="AI Study Assistant API")

# CORS (if you build a frontend elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
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
    """
    Upload one or more PDFs and rebuild the vector index.
    """
    reset_collection()
    docs = build_doc_chunks(files)
    add_documents_to_chroma(docs)
    return {"message": "Index built", "chunks": len(docs)}


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Ask a question based on the currently indexed PDFs.
    """
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set on server."}

    answer, context, chunks = query_study_notes(req.question, req.k)
    return {
        "question": req.question,
        "answer": answer,
        "context": context,
        "chunks": chunks,
    }
