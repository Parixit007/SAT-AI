"""The land-cover specialist's adapter (specialists/land_cover_adapter.py): the plain-language summary and its
stated limits, the JSON-able facts, registration, and the handler with the model stubbed out."""

import numpy as np
import pytest
from PIL import Image

from app.config import MODELS_DIR
from app.orchestrator.tool_registry import QueryInput
from app.specialists import DEFAULT_REGISTRY, land_cover_adapter as adapter
from app.specialists._loader import load_module

FACTS = {
    "fractions": {"bareland": 0.004, "rangeland": 0.05, "developed space": 0.15, "road": 0.18, "tree": 0.22,
                  "water": 0.0, "agriculture land": 0.0, "building": 0.396},
    "dominant": "building", "building_count": 62,
    "roof_colors": [{"name": "white", "share": 0.41, "rgb": [231, 228, 222]}, {"name": "grey", "share": 0.28, "rgb": [120, 120, 120]},
                    {"name": "red", "share": 0.12, "rgb": [190, 60, 50]}, {"name": "brown", "share": 0.06, "rgb": [110, 70, 40]}],
    "confidence": 0.83, "image_size": [1024, 1024],
}


def test_the_summary_lists_classes_largest_first_and_leaves_out_specks():
    text = adapter.summarize(FACTS)
    assert text.index("building 40%") < text.index("tree 22%") < text.index("road 18%") < text.index("developed space 15%") < text.index("rangeland 5%")
    assert "bareland" not in text and "water" not in text  # under 1% of the image


def test_the_summary_states_the_limits_of_the_building_count_and_roof_colours():
    text = adapter.summarize(FACTS)
    assert "About 62 separate building outlines" in text and "dense blocks are under-counted" in text
    assert "Roofs are mostly white (41%), grey (28%) and red (12%)" in text  # top three; brown (6%) is left out
    assert "shadows can look like dark roofs" in text


def test_no_buildings_says_so_and_skips_the_roofs():
    facts = {**FACTS, "fractions": {**FACTS["fractions"], "building": 0.0, "tree": 0.9}, "building_count": 0, "roof_colors": []}
    text = adapter.summarize(facts)
    assert "No buildings were found." in text and "Roofs" not in text


def test_buildings_without_separable_outlines_are_not_reported_as_none():
    facts = {**FACTS, "building_count": 0}
    assert "no separate building outlines could be told apart" in adapter.summarize(facts)


def test_the_tool_is_used_only_when_switched_on_and_installed(tmp_path):
    checkpoint = tmp_path / "landcover_unet.pt"
    assert adapter.landcover_available(True, checkpoint) is False   # switched on, nothing installed
    checkpoint.write_bytes(b"x")
    assert adapter.landcover_available(True, checkpoint) is True
    assert adapter.landcover_available(False, checkpoint) is False  # installed, switched off


def test_it_is_registered_as_a_single_optical_image_tool_that_says_what_it_does_not_do():
    spec = DEFAULT_REGISTRY.get("land_cover_analysis")
    assert spec is not None and (spec.min_images, spec.max_images) == (1, 1) and not spec.requires_location
    assert "does NOT count cars" in spec.description and "scene_description" in spec.description


class _StubTool:
    def analyze(self, image_path):
        class_map = np.full((40, 60), 7, dtype=np.uint8)
        class_map[:, 30:] = 4
        return {"class_map": class_map, "fractions": {"building": 0.5123456, "tree": 0.4876544, **{n: 0.0 for n in
                ["bareland", "rangeland", "developed space", "road", "water", "agriculture land"]}},
                "building_count": 3, "roof_colors": [{"name": "white", "share": 1.0, "rgb": [240, 240, 240]}],
                "confidence": 0.912345, "image_size": [60, 40]}


def test_analyze_returns_json_able_facts_with_rounded_numbers(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_get_tool", lambda: _StubTool())
    facts, class_map = adapter.analyze("ignored.png")
    assert class_map.shape == (40, 60)
    assert facts["dominant"] == "building" and facts["building_count"] == 3
    assert facts["fractions"]["building"] == 0.5123 and facts["confidence"] == 0.9123
    import json
    json.dumps(facts)  # must serialise as-is: it goes straight into the API response


def test_the_handler_returns_the_summary_the_facts_and_an_overlay(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_get_tool", lambda: _StubTool())
    monkeypatch.setattr(adapter, "EVIDENCE_DIR", tmp_path)
    scene = tmp_path / "scene.png"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(scene)

    result = adapter._handle(QueryInput(images=[scene]), {})

    assert result.tool_name == "land_cover_analysis" and result.confidence == 0.9123
    assert "building 51%" in result.text_summary and result.structured_data["building_count"] == 3
    assert result.evidence_image_path.exists() and result.evidence_image_path.parent == tmp_path
