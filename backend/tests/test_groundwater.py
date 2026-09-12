"""Tests for the groundwater-potential tool, split to match its own internal split:
- compute_groundwater_score() is pure Python -- fully tested here, no GEE involved.
- fetch_raw_layer_values()/render_thumbnail_url() need live Earth Engine credentials -- this suite
  doesn't assume they're configured (CI/other machines won't have them), so those two are covered
  by a monkeypatched end-to-end routing test plus a test that "not configured" fails cleanly. A
  live GEE service account exists in this dev environment and has been used for ad-hoc manual
  verification against real locations, but that isn't re-run automatically here."""

import pytest

from app.gee.groundwater import LAND_COVER_SUITABILITY, compute_groundwater_score
from app.orchestrator.controller import handle_query
from app.orchestrator.llm_providers.base import ToolCall
from app.orchestrator.tool_registry import LatLon, QueryInput
from app.specialists import build_default_registry
from tests.conftest import StubProvider

# --- compute_groundwater_score: pure logic, fully testable without GEE ----------------------


def test_all_favorable_inputs_score_very_high():
    # cropland's own suitability constant is 0.65 (no land-cover class maps to a full 1.0 -- by
    # design, see LAND_COVER_SUITABILITY), so max achievable overall here is
    # 0.30*1 + 0.30*1 + 0.20*1 + 0.20*0.65 = 0.93, not 1.0.
    raw = {"rainfall": 2500.0, "twi": 15.0, "dist_to_water": 0.0, "landcover": 40}  # cropland
    score = compute_groundwater_score(raw)

    assert score.category == "Very High"
    assert score.overall_score == pytest.approx(0.93, abs=0.01)
    assert score.missing_layers == []


def test_all_unfavorable_inputs_score_very_low():
    # built-up's suitability constant is 0.1 (not 0.0), so overall = 0.20*0.1 = 0.02, not exactly 0.
    raw = {"rainfall": 0.0, "twi": 2.0, "dist_to_water": 5000.0, "landcover": 50}  # built-up
    score = compute_groundwater_score(raw)

    assert score.category == "Very Low"
    assert score.overall_score == pytest.approx(0.02, abs=0.01)


def test_missing_layer_is_excluded_not_zeroed():
    complete = compute_groundwater_score({"rainfall": 2500.0, "twi": 15.0, "dist_to_water": 0.0, "landcover": 40})
    missing_rainfall = compute_groundwater_score({"rainfall": None, "twi": 15.0, "dist_to_water": 0.0, "landcover": 40})

    assert "rainfall" in missing_rainfall.missing_layers
    # A missing layer should be excluded (weights renormalized), not scored as 0 -- so the
    # remaining-favorable-everything case should still land in the same top category, not be
    # dragged down as if rainfall were terrible.
    assert missing_rainfall.category == complete.category == "Very High"


def test_landcover_floating_point_noise_from_ee_reducer_is_not_dropped():
    # Regression test: a live Earth Engine mode-reducer returned 49.999999999999996 for what's
    # semantically class 50 -- int() truncates that to 49 (a lookup miss), round() correctly gets 50.
    score = compute_groundwater_score(
        {"rainfall": 1000.0, "twi": 8.0, "dist_to_water": 1000.0, "landcover": 49.999999999999996}
    )
    assert "landcover" not in score.missing_layers
    assert score.layers["landcover"].normalized == LAND_COVER_SUITABILITY[50]


def test_all_layers_missing_scores_zero_not_a_crash():
    score = compute_groundwater_score({"rainfall": None, "twi": None, "dist_to_water": None, "landcover": None})

    assert score.overall_score == 0.0
    assert score.category == "Very Low"
    assert set(score.missing_layers) == {"rainfall", "twi", "dist_to_water", "landcover"}


def test_unknown_landcover_class_code_is_treated_as_missing():
    # A WorldCover code not in LAND_COVER_SUITABILITY (shouldn't normally happen, but the lookup
    # must degrade gracefully rather than KeyError).
    score = compute_groundwater_score({"rainfall": 1000.0, "twi": 8.0, "dist_to_water": 1000.0, "landcover": 999})

    assert "landcover" in score.missing_layers


def test_values_at_range_boundaries_do_not_error():
    # Rainfall/dist_to_water beyond their normalization range should clip to [0,1], not error.
    score = compute_groundwater_score({"rainfall": 10_000.0, "twi": -5.0, "dist_to_water": -100.0, "landcover": 10})
    assert 0.0 <= score.overall_score <= 1.0


# --- End-to-end routing through the real adapter, with GEE calls monkeypatched --------------


def test_groundwater_tool_is_registered():
    registry = build_default_registry()
    spec = registry.get("groundwater_potential")

    assert spec is not None
    assert spec.requires_location is True
    assert spec.uses_images is False


def test_groundwater_handler_raises_clean_error_without_gee_configured(monkeypatch):
    # Explicitly simulate "unconfigured" rather than relying on ambient .env state -- a real GEE
    # service account may well be configured on whatever machine runs this suite (it is in dev),
    # and this test needs to be deterministic regardless.
    import app.gee.client as client_module

    monkeypatch.setattr(client_module.settings, "gee_service_account_email", None)
    monkeypatch.setattr(client_module, "_initialized", False)

    from app.orchestrator.tool_registry import QueryInput
    from app.specialists.groundwater_adapter import _handle

    with pytest.raises(RuntimeError, match="Google Earth Engine is not configured"):
        _handle(QueryInput(location=LatLon(lat=12.9, lon=77.6)), {})


def test_groundwater_routes_end_to_end_with_gee_mocked(monkeypatch):
    import app.specialists.groundwater_adapter as adapter

    monkeypatch.setattr(
        adapter,
        "assess_groundwater_potential",
        lambda lat, lon, radius_m: compute_groundwater_score(
            {"rainfall": 2000.0, "twi": 12.0, "dist_to_water": 200.0, "landcover": 40}
        ),
    )
    def _boom(lat, lon, radius_m):
        raise RuntimeError("thumbnail rendering unavailable in this test")

    monkeypatch.setattr(adapter, "render_thumbnail_url", _boom)

    registry = build_default_registry()
    provider = StubProvider([ToolCall(tool_name="groundwater_potential", arguments={})])

    result = handle_query(
        "should I dig a well here?", QueryInput(location=LatLon(lat=12.9, lon=77.6)), provider, registry
    )

    assert result.trace.selected_task == "groundwater_potential"
    assert "High" in result.answer_text or "Moderate" in result.answer_text
    # thumbnail rendering was made to fail on purpose -- must degrade gracefully, not sink the result
    assert result.trace.tools_used[0]["name"] == "groundwater_potential"
    assert 0.0 <= result.confidence <= 1.0
