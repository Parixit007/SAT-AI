import pytest
from fastapi.testclient import TestClient

import app.api.routes_query as routes_query
from app.main import app
from app.orchestrator.llm_providers.base import LLMProvider, ToolCall
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


def test_upload_rejects_path_traversal_filename(client, sample_image):
    """Regression test: `f.filename` used to be joined into the destination path unsanitized, so a
    crafted filename with `../` segments (or an absolute path -- pathlib's `/` would discard the
    upload dir entirely) could write outside the per-upload directory."""
    from app.config import UPLOADS_DIR

    with open(sample_image, "rb") as f:
        r = client.post(
            "/api/upload",
            files={"files": ("../../../../tmp/evil_traversal.png", f, "image/png")},
        )
    assert r.status_code == 200
    input_id = r.json()["input_id"]

    # The one file written landed inside this upload's own directory, with no directory
    # components from the crafted filename surviving into the saved name.
    saved = list((UPLOADS_DIR / input_id).iterdir())
    assert len(saved) == 1
    assert saved[0].parent == UPLOADS_DIR / input_id
    assert ".." not in saved[0].name and "/" not in saved[0].name


def test_upload_same_filename_twice_does_not_overwrite(client, sample_image):
    """Regression test: two files sharing a filename in one batch (e.g. an optical+SAR pair both
    called 'export.tif') used to silently overwrite each other on disk."""
    with open(sample_image, "rb") as f1, open(sample_image, "rb") as f2:
        r = client.post(
            "/api/upload",
            files=[
                ("files", ("export.tif", f1, "image/tiff")),
                ("files", ("export.tif", f2, "image/tiff")),
            ],
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["images"]) == 2
    stored_paths = {img["stored_path"] for img in body["images"]}
    assert len(stored_paths) == 2  # both files actually exist on disk, distinctly


def test_upload_partial_failure_does_not_poison_the_whole_input_id(client, sample_image):
    """Regression test: one bad file used to be persisted alongside the good ones (unfiltered),
    so every future query against that input_id failed input validation entirely -- even when
    other files in the same batch were perfectly valid."""
    with open(sample_image, "rb") as good:
        r = client.post(
            "/api/upload",
            files=[
                ("files", ("good.png", good, "image/png")),
                ("files", ("bad.txt", b"not an image", "text/plain")),
            ],
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["images"]) == 1  # only the good file was saved
    assert len(body["errors"]) == 1  # the bad one is reported distinctly, not silently dropped
    input_id = body["input_id"]

    r = client.post("/api/query", json={"input_id": input_id, "query_text": "how much water?"})
    assert r.status_code == 200
    assert r.json()["execution_trace"]["selected_task"] != "input_validation_failed"


def test_query_falls_back_to_uploaded_images_own_geo_metadata(client, georeferenced_tif):
    """_resolve_location's image-geo fallback (routes_query.py) had no test at all -- every
    existing case here either supplies an explicit `location` or uploads a non-georeferenced
    image. `input_id` being optional and location-only queries needing no upload are both
    documented behaviors (CLAUDE.md); this is the other half -- an upload supplying the location
    on its own, with no explicit `location` in the request at all."""
    with open(georeferenced_tif, "rb") as f:
        r = client.post("/api/upload", files={"files": ("geo.tif", f, "image/tiff")})
    assert r.status_code == 200
    body = r.json()
    input_id = body["input_id"]
    geo = body["images"][0]["geo"]
    assert geo is not None  # sanity: the fixture is actually georeferenced

    r = client.post("/api/query", json={"input_id": input_id, "query_text": "water?"})
    assert r.status_code == 200
    input_summary = r.json()["execution_trace"]["input_summary"]
    assert f"{geo['center_lat']:.5f}" in input_summary
    assert f"{geo['center_lon']:.5f}" in input_summary


def test_query_returns_503_when_llm_provider_fails(monkeypatch, sample_image):
    """The LLM-provider-failure -> clean 503 path (CLAUDE.md's other deliberate failure-handling
    layer, alongside specialist failures -> trace warnings) had no test at all -- the `client`
    fixture above always monkeypatches a working StubProvider, so this never got exercised."""

    class BrokenProvider(LLMProvider):
        def select_tools(self, query, tool_specs, input_summary):
            raise RuntimeError("Groq request failed (some-model): 401 unauthorized")

    monkeypatch.setattr(routes_query, "get_provider", lambda name: BrokenProvider())
    broken_client = TestClient(app)

    with open(sample_image, "rb") as f:
        r = broken_client.post("/api/upload", files={"files": ("scene.png", f, "image/png")})
    input_id = r.json()["input_id"]

    r = broken_client.post("/api/query", json={"input_id": input_id, "query_text": "water?"})
    assert r.status_code == 503


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
