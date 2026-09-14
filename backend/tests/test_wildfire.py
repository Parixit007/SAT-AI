"""Tests for the wildfire-detection tool, split to match its own internal split (same pattern as
test_groundwater.py):
- classify_fire_activity() is pure Python -- fully tested here, no GEE involved.
- fetch_recent_fire_activity()/render_thumbnail_url() need live Earth Engine credentials -- this
  suite doesn't assume they're configured (CI/other machines won't have them), so those two are
  covered by a monkeypatched end-to-end routing test plus a test that "not configured" fails
  cleanly. A live GEE service account exists in this dev environment and was used for ad-hoc
  verification against a real detected fire pixel (western US, Sept 2026) -- see wildfire.py's
  own module docstring -- but that isn't re-run automatically here."""

import pytest

from app.gee.wildfire import HIGH_CONFIDENCE_THRESHOLD, classify_fire_activity
from app.orchestrator.controller import handle_query
from app.orchestrator.llm_providers.base import ToolCall
from app.orchestrator.tool_registry import LatLon, QueryInput
from app.specialists import build_default_registry
from tests.conftest import StubProvider

# --- classify_fire_activity: pure logic, fully testable without GEE -------------------------


def test_no_hits_means_no_fire_detected():
    assessment = classify_fire_activity([], lookback_days=10)

    assert assessment.status == "No Fire Detected"
    assert assessment.detection_days == 0
    assert assessment.max_confidence is None
    assert assessment.max_brightness_k is None
    assert assessment.most_recent_date is None
    assert assessment.hits == []


def test_high_confidence_hit_is_active_fire_detected():
    hits = [{"date": "2026-09-10", "max_confidence": HIGH_CONFIDENCE_THRESHOLD, "max_brightness_k": 330.0}]
    assessment = classify_fire_activity(hits, lookback_days=10)

    assert assessment.status == "Active Fire Detected"
    assert assessment.max_confidence == HIGH_CONFIDENCE_THRESHOLD


def test_low_confidence_hit_is_possible_fire_activity_not_active():
    hits = [{"date": "2026-09-10", "max_confidence": HIGH_CONFIDENCE_THRESHOLD - 1, "max_brightness_k": 310.0}]
    assessment = classify_fire_activity(hits, lookback_days=10)

    assert assessment.status == "Possible Fire Activity"


def test_most_recent_date_and_peak_values_picked_across_multiple_hits():
    hits = [
        {"date": "2026-09-02", "max_confidence": 79.0, "max_brightness_k": 325.16},
        {"date": "2026-09-08", "max_confidence": 95.0, "max_brightness_k": 340.0},
        {"date": "2026-09-05", "max_confidence": 60.0, "max_brightness_k": 315.0},
    ]
    assessment = classify_fire_activity(hits, lookback_days=10)

    assert assessment.detection_days == 3
    assert assessment.most_recent_date == "2026-09-08"
    assert assessment.max_confidence == 95.0
    assert assessment.max_brightness_k == 340.0
    # hits should come back sorted chronologically, not in input order
    assert [h.date for h in assessment.hits] == ["2026-09-02", "2026-09-05", "2026-09-08"]


def test_lookback_days_is_carried_through_unchanged():
    assessment = classify_fire_activity([], lookback_days=21)
    assert assessment.lookback_days == 21


# --- End-to-end routing through the real adapter, with GEE calls monkeypatched --------------


def test_wildfire_tool_is_registered():
    registry = build_default_registry()
    spec = registry.get("wildfire_detection")

    assert spec is not None
    assert spec.requires_location is True
    assert spec.uses_images is False


def test_wildfire_handler_raises_clean_error_without_gee_configured(monkeypatch):
    # Explicitly simulate "unconfigured" rather than relying on ambient .env state -- a real GEE
    # service account may well be configured on whatever machine runs this suite (it is in dev),
    # and this test needs to be deterministic regardless.
    import app.gee.client as client_module

    monkeypatch.setattr(client_module.settings, "gee_service_account_email", None)
    monkeypatch.setattr(client_module, "_initialized", False)

    from app.specialists.wildfire_adapter import _handle

    with pytest.raises(RuntimeError, match="Google Earth Engine is not configured"):
        _handle(QueryInput(location=LatLon(lat=38.5, lon=-120.5)), {})


def test_wildfire_routes_end_to_end_with_gee_mocked(monkeypatch):
    import app.specialists.wildfire_adapter as adapter
    from app.gee.wildfire import classify_fire_activity as real_classify

    monkeypatch.setattr(
        adapter,
        "assess_wildfire_risk",
        lambda lat, lon, radius_m, lookback_days: real_classify(
            [{"date": "2026-09-12", "max_confidence": 90.0, "max_brightness_k": 350.0}], lookback_days
        ),
    )

    def _boom(lat, lon, radius_m, lookback_days):
        raise RuntimeError("thumbnail rendering unavailable in this test")

    monkeypatch.setattr(adapter, "render_thumbnail_url", _boom)

    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="wildfire_detection", arguments={})])

    result = handle_query(
        "is there a fire near here?", QueryInput(location=LatLon(lat=38.5, lon=-120.5)), provider, registry
    )

    assert result.trace.selected_task == "wildfire_detection"
    assert "Active Fire Detected" in result.answer_text
    # thumbnail rendering was made to fail on purpose -- must degrade gracefully, not sink the result
    assert result.trace.tools_used[0]["name"] == "wildfire_detection"
    assert 0.0 <= result.confidence <= 1.0


def test_wildfire_routes_end_to_end_with_no_detection(monkeypatch):
    import app.specialists.wildfire_adapter as adapter
    from app.gee.wildfire import classify_fire_activity as real_classify

    monkeypatch.setattr(
        adapter, "assess_wildfire_risk", lambda lat, lon, radius_m, lookback_days: real_classify([], lookback_days)
    )

    def _boom(lat, lon, radius_m, lookback_days):
        raise RuntimeError("thumbnail rendering unavailable in this test")

    monkeypatch.setattr(adapter, "render_thumbnail_url", _boom)

    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="wildfire_detection", arguments={})])

    result = handle_query(
        "any wildfires near this spot?", QueryInput(location=LatLon(lat=10.0, lon=20.0)), provider, registry
    )

    assert "No Fire Detected" in result.answer_text
    assert result.trace.tools_used[0]["name"] == "wildfire_detection"
