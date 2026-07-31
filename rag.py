"""
rag.py

Retrieval-Augmented Generation: index study materials (PDF/TXT/MD) and
retrieve the most relevant chunks for a question, so answers can be
grounded in a student's own course material instead of just whatever the
model already knows.

Storage is a simple, custom SQLite store -- the same local database file
memory.py uses (memory.db by default). Chunks and their embeddings live in
plain tables, and retrieval is brute-force cosine similarity computed in
Python. No vector database dependency; this is fine at the scale of one
student's personal notes (dozens to low hundreds of chunks), and keeps the
project's "no server, one local file" approach consistent.

Embeddings come from a local Ollama embedding model via
llm_client.OllamaClient.embed() (e.g. nomic-embed-text) -- same local
stack as chat completions, just a different model.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from config import Settings
from llm_client import OllamaClient

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - only hit if pypdf isn't installed
    PdfReader = None

_DB_PATH = Settings().memory_db_path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class Chunk:
    id: int
    document_id: int
    filename: str
    content: str
    score: float = 0.0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            added_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id),
            user_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        );
        """
    )


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def extract_text(path: Path) -> str:
    """Read a .pdf, .txt, or .md file and return its plain text."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError(
                "Reading PDFs needs pypdf: pip install pypdf (see requirements.txt)"
            )
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character chunks. Simple and
    dependency-free -- doesn't try to respect sentence or paragraph
    boundaries, just keeps chunks a reasonable size for embedding and
    gives neighbors some shared context via the overlap.
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def ingest_file(client: OllamaClient, user_id: int, path: Path) -> int:
    """
    Extract, chunk, embed, and store a study material file for this user.
    Returns the number of chunks stored (0 if no text could be extracted).
    """
    text = extract_text(path)
    pieces = chunk_text(text)
    if not pieces:
        return 0

    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO documents (user_id, filename, added_at) VALUES (?, ?, ?)",
            (user_id, path.name, _now()),
        )
        document_id = cursor.lastrowid

        for index, piece in enumerate(pieces):
            embedding = client.embed(piece)
            conn.execute(
                "INSERT INTO chunks "
                "(document_id, user_id, chunk_index, content, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (document_id, user_id, index, piece, json.dumps(embedding)),
            )
        conn.commit()

    return len(pieces)


def retrieve(
    client: OllamaClient, user_id: int, query: str, top_k: int = DEFAULT_TOP_K
) -> list[Chunk]:
    """
    Embed `query` and return the top_k most similar chunks for this user,
    highest similarity first. Empty list if nothing has been indexed yet.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT chunks.id, chunks.document_id, chunks.content, chunks.embedding, "
            "documents.filename "
            "FROM chunks JOIN documents ON documents.id = chunks.document_id "
            "WHERE chunks.user_id = ?",
            (user_id,),
        ).fetchall()

    if not rows:
        return []

    query_embedding = client.embed(query)

    scored = [
        Chunk(
            id=row["id"],
            document_id=row["document_id"],
            filename=row["filename"],
            content=row["content"],
            score=_cosine_similarity(query_embedding, json.loads(row["embedding"])),
        )
        for row in rows
    ]

    scored.sort(key=lambda chunk: chunk.score, reverse=True)
    return scored[:top_k]


def list_documents(user_id: int) -> list[tuple[str, str]]:
    """Return (filename, added_at) for this user's indexed documents, most recent first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename, added_at FROM documents WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
    return [(row["filename"], row["added_at"]) for row in rows]


def build_rag_messages(chunks: list[Chunk], question: str) -> list[dict]:
    """
    Build a system+user message pair that grounds the model's answer in
    the retrieved chunks, with explicit instructions to say so rather than
    guess if the material doesn't actually cover the question.
    """
    context = "\n\n".join(f"[From {chunk.filename}]\n{chunk.content}" for chunk in chunks)
    system = (
        "You are an AI tutor. Below is context retrieved from the student's "
        "own study material -- use it as your source of truth, but do not "
        "just quote or list it back. Write a real answer: understand what "
        "the context says, then explain it in your own words, clearly and "
        "completely, as a tutor would -- define terms, connect ideas, add a "
        "short example if it helps understanding. If the context only "
        "partly covers the question, answer the part it covers well and say "
        "what's missing. If it doesn't cover the question at all, say so "
        "plainly instead of guessing."
    )
    user = (
        f"Context from study materials:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question directly and in full -- don't just restate or "
        "summarize the context snippets."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
