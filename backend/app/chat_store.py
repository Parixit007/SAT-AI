"""Persistent chat history -- unlike store.py's in-process dict (input_id -> paths, query_id ->
result, gone on restart), a *chat* is meant to survive across backend restarts and browser
sessions, so it needs a real file-backed store. SQLite (stdlib, no new dependency) at
`config.CHAT_DB_PATH` -- more than enough for a single-process demo app, and the same "small,
dependency-free store" spirit as everything else in this codebase.

One connection per call (`_connect()`), not a shared module-level connection: `routes_query.py`'s
query handling runs inside `run_in_threadpool`, so concurrent requests can land on different
threads, and a bare `sqlite3.Connection` isn't safe to share across threads. Opening a short-lived
connection per call is the standard, simplest-to-reason-about way to use SQLite from a small
multi-threaded app, and query volume here is nowhere near where that would be a real cost.

A chat is a list of entries in order; each entry stores the *entire* QueryResponse the API already
returns for a live query, JSON-serialized verbatim -- so a history entry and a live result are
exactly the same shape on the wire, and the frontend needs no second rendering path for either."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from app.config import CHAT_DB_PATH

TITLE_MAX_LEN = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_entries (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_entries_chat_id ON chat_entries(chat_id, seq);
        """
    )


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHAT_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        _init_schema(conn)  # CREATE TABLE IF NOT EXISTS -- cheap, idempotent, avoids a separate startup step
        yield conn
        conn.commit()
    finally:
        conn.close()


def _title_from_query(query_text: str) -> str:
    text = " ".join(query_text.split())  # collapse newlines/repeated whitespace for a clean title
    if len(text) <= TITLE_MAX_LEN:
        return text or "Untitled chat"
    return text[:TITLE_MAX_LEN].rstrip() + "…"


def create_chat(first_query_text: str) -> str:
    chat_id = uuid.uuid4().hex
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, _title_from_query(first_query_text), now, now),
        )
    return chat_id


def chat_exists(chat_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return row is not None


def add_entry(chat_id: str, query_text: str, response_json: str) -> None:
    now = _now()
    with _connect() as conn:
        next_seq = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_entries WHERE chat_id = ?", (chat_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO chat_entries (id, chat_id, seq, query_text, response_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, chat_id, next_seq, query_text, response_json, now),
        )
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))


def list_chats() -> list[dict[str, Any]]:
    """Most-recently-active first -- the order a chat-history list is expected to read in."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(e.id) AS entry_count
            FROM chats c LEFT JOIN chat_entries e ON e.chat_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_chat(chat_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        chat = conn.execute("SELECT id, title, created_at, updated_at FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if chat is None:
            return None
        entries = conn.execute(
            "SELECT query_text, response_json, created_at FROM chat_entries WHERE chat_id = ? ORDER BY seq ASC",
            (chat_id,),
        ).fetchall()
    return {
        **dict(chat),
        "entries": [
            {"query_text": e["query_text"], "created_at": e["created_at"], "response": json.loads(e["response_json"])}
            for e in entries
        ],
    }


def delete_chat(chat_id: str) -> bool:
    """True if a chat was actually deleted, False if chat_id didn't exist -- lets the route return
    a clean 404 instead of a silent no-op. Note: this only removes the DB rows -- any evidence/
    upload image files the chat's entries reference stay on disk, same as they always have (this
    app has never garbage-collected those; see store.py's own docstring)."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.execute("DELETE FROM chat_entries WHERE chat_id = ?", (chat_id,))
        return cur.rowcount > 0
