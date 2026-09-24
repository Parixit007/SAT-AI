import pytest
from fastapi.testclient import TestClient

import app.api.routes_query as routes_query
from app.main import app
from app.orchestrator.llm_providers.base import ToolCall
from tests.conftest import StubProvider

# No isolation fixture for the chat DB (matching this codebase's established convention --
# test_api.py's own docstring-level comment: uploads/evidence dirs aren't isolated either).
# Every test below asserts on the presence/absence of the specific chat_id(s) it created, never on
# total counts, so accumulated rows from other test runs against the same data/chat_history.db
# can't make these flaky.


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes_query,
        "get_provider",
        lambda name: StubProvider([ToolCall(tool_name="water_body_segmentation", arguments={})]),
    )
    return TestClient(app)


def _query(client, sample_image, query_text="how much water?", chat_id=None):
    with open(sample_image, "rb") as f:
        r = client.post("/api/upload", files={"files": ("scene.png", f, "image/png")})
    input_id = r.json()["input_id"]
    payload = {"input_id": input_id, "query_text": query_text}
    if chat_id is not None:
        payload["chat_id"] = chat_id
    return client.post("/api/query", json=payload)


def test_query_with_no_chat_id_creates_a_new_chat(client, sample_image):
    r = _query(client, sample_image, "how much water is here?")
    assert r.status_code == 200
    chat_id = r.json()["chat_id"]
    assert chat_id  # a real id was minted, not left blank

    r = client.get(f"/api/chats/{chat_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["title"] == "how much water is here?"
    assert len(detail["entries"]) == 1
    assert detail["entries"][0]["query_text"] == "how much water is here?"
    # The saved entry's response is the exact QueryResponse the live call returned.
    assert detail["entries"][0]["response"]["chat_id"] == chat_id


def test_query_with_existing_chat_id_appends_an_entry(client, sample_image):
    first = _query(client, sample_image, "first question")
    chat_id = first.json()["chat_id"]

    second = _query(client, sample_image, "second question", chat_id=chat_id)
    assert second.status_code == 200
    assert second.json()["chat_id"] == chat_id  # same chat, not a new one

    detail = client.get(f"/api/chats/{chat_id}").json()
    assert [e["query_text"] for e in detail["entries"]] == ["first question", "second question"]
    # Title stays from the first query -- appending doesn't retitle the chat.
    assert detail["title"] == "first question"


def test_query_with_unknown_chat_id_is_404_and_creates_nothing(client, sample_image):
    r = _query(client, sample_image, chat_id="does-not-exist")
    assert r.status_code == 404

    r = client.get("/api/chats")
    assert all(c["id"] != "does-not-exist" for c in r.json())


def test_list_chats_includes_a_created_chat_most_recent_first(client, sample_image):
    r = _query(client, sample_image, "list me please")
    chat_id = r.json()["chat_id"]

    r = client.get("/api/chats")
    assert r.status_code == 200
    chats = r.json()
    matches = [c for c in chats if c["id"] == chat_id]
    assert len(matches) == 1
    assert matches[0]["title"] == "list me please"
    assert matches[0]["entry_count"] == 1
    # Most-recently-updated chat should be at (or very near) the front of the list.
    assert chats[0]["id"] == chat_id


def test_get_unknown_chat_is_404(client):
    r = client.get("/api/chats/does-not-exist")
    assert r.status_code == 404


def test_delete_chat_removes_it(client, sample_image):
    r = _query(client, sample_image, "delete me")
    chat_id = r.json()["chat_id"]

    r = client.delete(f"/api/chats/{chat_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    r = client.get(f"/api/chats/{chat_id}")
    assert r.status_code == 404

    r = client.get("/api/chats")
    assert all(c["id"] != chat_id for c in r.json())

    # A second delete of the same (now-gone) chat is a clean 404, not a silent no-op success.
    r = client.delete(f"/api/chats/{chat_id}")
    assert r.status_code == 404


def test_delete_unknown_chat_is_404(client):
    r = client.delete("/api/chats/does-not-exist")
    assert r.status_code == 404
