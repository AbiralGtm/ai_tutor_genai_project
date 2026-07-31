"""
memory.py

Memory System: user identity + learning history, persisted in a local
SQLite database (sqlite3 is in Python's standard library -- no new
dependency). This module has three jobs:

1. Identity   -- login() turns a typed name into a persistent user profile
                 (users table), reused across separate runs of the app.
2. History    -- record_interaction() logs every template/technique or
                 pipeline run under that profile (interactions table);
                 recent_history() reads it back. For chat.py's Prompting
                 Console, the full message transcript is stored too, so a
                 past conversation can be reloaded and continued verbatim
                 rather than just remembered as a topic.
3. Context    -- build_context() turns recent history into a short,
                 natural-language paragraph that chat.py and chains.py
                 inject into their prompts, so responses can actually
                 reflect what's known about the student.

Both chat.py and chains.py accept an optional `user` so a single login in
main.py's launcher can be shared across tools; each also falls back to its
own login() when run standalone.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from config import Settings

_DB_PATH = Settings().memory_db_path


@dataclass(frozen=True)
class User:
    id: int
    name: str
    created_at: str
    last_seen_at: str


@dataclass(frozen=True)
class Interaction:
    id: int
    user_id: int
    created_at: str
    tool: str
    activity: str
    topic: str | None
    technique: str | None
    summary: str
    transcript: list[dict] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL COLLATE NOCASE UNIQUE,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            created_at TEXT NOT NULL,
            tool TEXT NOT NULL,
            activity TEXT NOT NULL,
            topic TEXT,
            technique TEXT,
            summary TEXT NOT NULL
        );
        """
    )
    # Added after the initial release -- a plain CREATE TABLE IF NOT EXISTS
    # won't retrofit this column onto a database created before it existed,
    # so add it explicitly and ignore the error if it's already there.
    try:
        conn.execute("ALTER TABLE interactions ADD COLUMN transcript TEXT")
    except sqlite3.OperationalError:
        pass


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def get_or_create_user(name: str) -> tuple[User, bool]:
    """
    Look up a user by name (case-insensitive) or create a new profile.

    Returns (user, is_new). The returned user.last_seen_at is the
    timestamp from *before* this call -- i.e. "last seen" as of their
    previous session -- which is what a login greeting wants to show. The
    stored row is updated to the current time as a side effect.
    """
    now = _now()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()

        if row is None:
            cursor = conn.execute(
                "INSERT INTO users (name, created_at, last_seen_at) VALUES (?, ?, ?)",
                (name, now, now),
            )
            conn.commit()
            return User(id=cursor.lastrowid, name=name, created_at=now, last_seen_at=now), True

        previous_last_seen = row["last_seen_at"]
        conn.execute("UPDATE users SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
        conn.commit()
        user = User(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            last_seen_at=previous_last_seen,
        )
        return user, False


def record_interaction(
    user_id: int,
    tool: str,
    activity: str,
    topic: str | None,
    technique: str | None,
    summary: str,
    transcript: list[dict] | None = None,
) -> None:
    """
    Log one completed interaction to the user's learning history.

    `transcript`, if given, is the full list of chat messages (system/
    user/assistant dicts) for that conversation, stored as JSON so it can
    be reloaded later by "Continue a Conversation". Only chat.py's
    Prompting Console passes this; chains.py's pipeline runs don't have an
    equivalent ongoing conversation to resume.
    """
    with _connect() as conn:
        conn.execute(
            "INSERT INTO interactions "
            "(user_id, created_at, tool, activity, topic, technique, summary, transcript) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                _now(),
                tool,
                activity,
                topic,
                technique,
                summary,
                json.dumps(transcript) if transcript is not None else None,
            ),
        )
        conn.commit()


def recent_history(user_id: int, limit: int = 5) -> list[Interaction]:
    """Most recent interactions first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

    return [
        Interaction(
            id=row["id"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            tool=row["tool"],
            activity=row["activity"],
            topic=row["topic"],
            technique=row["technique"],
            summary=row["summary"],
            transcript=json.loads(row["transcript"]) if row["transcript"] else None,
        )
        for row in rows
    ]


def build_context(user: User, history: list[Interaction]) -> str:
    """
    Turn a user + their recent history into a short paragraph meant to be
    injected into a system prompt, so responses can be personalized.
    """
    if not history:
        return f"You are tutoring {user.name}. This is their first recorded session -- no prior learning history yet."

    seen: set[str] = set()
    topics: list[str] = []
    for h in history:
        if h.topic and h.topic not in seen:
            seen.add(h.topic)
            topics.append(h.topic)
    recap = "; ".join(f'"{t}"' for t in topics) if topics else "a few earlier sessions"
    return (
        f"You are tutoring {user.name}. They have recently studied: {recap}. "
        "Use this history to personalize your response where it's genuinely "
        "helpful (e.g. build on familiar topics, match their apparent level), "
        "but stay focused on answering the current request."
    )


def login() -> User | None:
    """
    Prompt for a name, resolve it to a persistent profile, and print a
    short greeting summarizing what's known so far. Returns None if the
    user quits (blank name after retry, Ctrl+C, or Ctrl+D).
    """
    print("\nWho's studying today?")
    while True:
        try:
            name = input("Enter your name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if name:
            break
        print("A name is required (or Ctrl+C to quit).")

    user, is_new = get_or_create_user(name)

    if is_new:
        print(f"Welcome, {user.name}! Your profile has been created.")
        return user

    print(f"Welcome back, {user.name}! Last seen {user.last_seen_at}.")
    history = recent_history(user.id, limit=3)
    topics = [h.topic for h in history if h.topic]
    if topics:
        print("Recently studied: " + "; ".join(f'"{t}"' for t in topics))

    return user
