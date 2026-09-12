import pytest
from fastapi.testclient import TestClient

import app.api.routes_query as routes_query
from app.main import app
from app.orchestrator.llm_providers.base import ToolCall
from tests.conftest import StubProvider


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes_query,
        "get_provider",
        lambda name: StubProvider([ToolCall(tool_name="water_body_segmentation", arguments={})]),
    )
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_upload_query_report_flow(client, sample_image):
    with open(sample_image, "rb") as f:
        r = client.post("/api/upload", files={"files": ("scene.png", f, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["images"][0]["format"] == "PNG"
    input_id = body["input_id"]

    r = client.post("/api/query", json={"input_id": input_id, "query_text": "how much water?"})
    assert r.status_code == 200
    query = r.json()
    assert query["execution_trace"]["selected_task"] == "water_body_segmentation"
    assert 0.0 <= query["confidence"] <= 1.0
    assert len(query["evidence_image_urls"]) == 1
    query_id = query["query_id"]

    r = client.get(f"/api/report/{query_id}")
    assert r.status_code == 200
    assert "water_body_segmentation" in r.text


def test_query_with_unknown_input_id_is_404(client):
    r = client.post("/api/query", json={"input_id": "does-not-exist", "query_text": "hi"})
    assert r.status_code == 404


def test_report_with_unknown_query_id_is_404(client):
    r = client.get("/api/report/does-not-exist")
    assert r.status_code == 404


def test_query_with_neither_input_id_nor_location_is_a_clean_validation_failure(client):
    # No 500/crash -- a query with nothing to work with is a valid (if unhelpful) auditable outcome.
    r = client.post("/api/query", json={"query_text": "hi"})
    assert r.status_code == 200
    assert r.json()["execution_trace"]["selected_task"] == "input_validation_failed"


def test_query_with_explicit_location_and_no_input_id_is_accepted(client):
    # No tool matches (registry has only image-based tools right now), but the request itself
    # -- location-only, no input_id/upload -- must be accepted and handled gracefully, not 404/500.
    r = client.post("/api/query", json={"query_text": "well here?", "location": {"lat": 12.9, "lon": 77.6}})
    assert r.status_code == 200
    assert "12.9" in r.json()["execution_trace"]["input_summary"]
