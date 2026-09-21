from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.orchestrator.tool_registry import QueryInput
from app.specialists import DEFAULT_REGISTRY, grounding_adapter, scene_description_adapter as adapter


class StubScanner:
    def __init__(self, detections):
        self.detections, self.calls = detections, []

    def scan(self, image_path, categories, box_threshold):
        self.calls.append({"categories": list(categories), "box_threshold": box_threshold})
        return self.detections


@pytest.fixture(autouse=True)
def no_captions_unless_asked(monkeypatch):
    """USE_CAPTIONS is decided at import by whether a gitignored checkpoint exists, so it differs
    between machines -- pin it off here and let the caption tests opt in via `with_captions`."""
    monkeypatch.setattr(adapter, "USE_CAPTIONS", False)
    monkeypatch.setattr(adapter, "USE_LANDCOVER", False)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "scene.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    return path


def _det(label, score, i=0):
    return {"phrase": label, "bbox_xyxy": [i, i, i + 8.0, i + 8.0], "score": score}


@pytest.mark.parametrize("enabled, installed, expected", [
    (True, True, True),
    (True, False, False),   # switched on but nothing to load: object scan only
    (False, True, False),   # installed but not switched on: must NOT quietly change what users are told
    (False, False, False),
])
def test_the_captioner_needs_both_the_switch_and_the_checkpoint(tmp_path, enabled, installed, expected):
    if installed:
        (tmp_path / "caption_meta.json").write_text("{}")
    assert adapter.captioner_available(enabled, tmp_path) is expected


def test_it_is_registered_as_a_single_image_tool():
    spec = DEFAULT_REGISTRY.get("scene_description")
    assert spec is not None and (spec.min_images, spec.max_images) == (1, 1) and not spec.requires_location


def test_it_counts_each_category_and_says_what_it_does_not_check(image, monkeypatch):
    # 4 airplanes (a cluster) and one ship strong enough to stand alone
    scanner = StubScanner([_det("airplane", 0.5, i) for i in range(3)] + [_det("airplane", 0.36, 9), _det("ship", 0.6)])
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: scanner)

    result = adapter._handle(QueryInput(images=[image]), {})

    assert scanner.calls == [{"categories": ["airplane", "ship", "storage tank"], "box_threshold": 0.30}]
    assert result.structured_data["objects"] == [
        {"label": "airplane", "count": 4, "best_score": 0.5},
        {"label": "ship", "count": 1, "best_score": 0.6},
    ]
    assert result.structured_data["total_objects"] == 5 and result.structured_data["unreported"] == []
    assert "checks only airplane, ship and storage tank" in result.text_summary
    assert "4 airplane(s); 1 ship(s)" in result.text_summary
    assert result.confidence == 0.6
    assert result.evidence_image_path is not None and result.evidence_image_path.exists()


@pytest.mark.parametrize("hits, reported", [
    ([0.39, 0.35], False),              # two "storage tanks" on Wembley Stadium
    ([0.32, 0.31, 0.30], False),        # three on a golf course: enough of them, but none strong enough
    ([0.40, 0.34, 0.32], True),         # a small cluster with one decent hit (Singapore's ships)
    ([0.30], False),                    # a lone weak hit
    ([0.56], True),                     # a lone hit strong enough to stand alone
    ([0.53, 0.36], False),              # a strong-looking pair is still only a pair (Flushing Meadows)
])
def test_the_precision_guard_holds_back_lone_and_paired_weak_detections(image, monkeypatch, hits, reported):
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([_det("storage tank", s, i) for i, s in enumerate(hits)]))

    result = adapter._handle(QueryInput(images=[image]), {})

    data = result.structured_data
    assert (data["total_objects"] == len(hits)) is reported
    assert (data["unreported"] == []) is reported
    if not reported:
        assert data["unreported"][0]["label"] == "storage tank" and data["unreported"][0]["count"] == len(hits)
        assert "none found" in result.text_summary and result.evidence_image_path is None and result.confidence == 0.0


def test_an_empty_scan_says_so_and_has_zero_confidence(image, monkeypatch):
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([]))
    result = adapter._handle(QueryInput(images=[image]), {})
    assert "none found" in result.text_summary
    assert result.confidence == 0.0 and result.evidence_image_path is None and result.structured_data["objects"] == []


# ---------------------------------------------------------------- the caption source

class StubCaptioner:
    def __init__(self, caption="The image shows a large airport apron with many parked aircraft.", confidence=0.6, error=None):
        self.result = {"caption": caption, "confidence": confidence}
        self.error, self.calls = error, 0

    def describe(self, image_path):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def with_captions(monkeypatch):
    monkeypatch.setattr(adapter, "USE_CAPTIONS", True)


def test_caption_and_object_scan_are_both_reported(image, monkeypatch, with_captions):
    captioner = StubCaptioner()
    monkeypatch.setattr(adapter, "_get_captioner", lambda: captioner)
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([_det("airplane", 0.5, i) for i in range(3)]))

    result = adapter._handle(QueryInput(images=[image]), {})

    assert result.structured_data["caption"].startswith("The image shows a large airport apron")
    assert result.structured_data["objects"] == [{"label": "airplane", "count": 3, "best_score": 0.5}]
    assert "Description (written by a small remote-sensing captioning model" in result.text_summary
    assert "found 3 airplane(s)" in result.text_summary
    assert result.confidence == pytest.approx((0.6 + 0.5) / 2)  # the mean of what the two sources reported


def test_a_failing_captioner_leaves_the_object_scan(image, monkeypatch, with_captions):
    monkeypatch.setattr(adapter, "_get_captioner", lambda: StubCaptioner(error=RuntimeError("no checkpoint")))
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([_det("ship", 0.4, i) for i in range(3)]))

    result = adapter._handle(QueryInput(images=[image]), {})

    assert result.structured_data["caption"] is None and result.structured_data["total_objects"] == 3
    assert "description model unavailable (no checkpoint)" in result.text_summary
    assert result.structured_data["notes"] == ["description model unavailable (no checkpoint)"]


def test_a_failing_scan_leaves_the_caption(image, monkeypatch, with_captions):
    class BrokenScanner:
        def scan(self, *args, **kwargs):
            raise RuntimeError("groundingdino missing")

    monkeypatch.setattr(adapter, "_get_captioner", lambda: StubCaptioner(confidence=0.7))
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: BrokenScanner())

    result = adapter._handle(QueryInput(images=[image]), {})

    assert result.structured_data["caption"] and result.structured_data["objects"] == []
    assert "object scan unavailable (groundingdino missing)" in result.text_summary
    assert result.confidence == pytest.approx(0.7)
    assert result.evidence_image_path is None


def test_when_both_sources_fail_the_tool_fails(image, monkeypatch, with_captions):
    class BrokenScanner:
        def scan(self, *args, **kwargs):
            raise RuntimeError("scan broke")

    monkeypatch.setattr(adapter, "_get_captioner", lambda: StubCaptioner(error=RuntimeError("caption broke")))
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: BrokenScanner())

    with pytest.raises(RuntimeError, match="scan broke.*caption broke|caption broke.*scan broke"):
        adapter._handle(QueryInput(images=[image]), {})


def test_the_captioner_is_not_touched_when_no_checkpoint_is_installed(image, monkeypatch):
    monkeypatch.setattr(adapter, "USE_CAPTIONS", False)
    captioner = StubCaptioner()
    monkeypatch.setattr(adapter, "_get_captioner", lambda: captioner)
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([]))

    result = adapter._handle(QueryInput(images=[image]), {})

    assert captioner.calls == 0 and result.structured_data["caption"] is None
    assert "Description" not in result.text_summary


# ---------------------------------------------------------------- the land-cover source

LAND_FACTS = {
    "fractions": {"bareland": 0.0, "rangeland": 0.05, "developed space": 0.15, "road": 0.18, "tree": 0.22,
                  "water": 0.0, "agriculture land": 0.0, "building": 0.40},
    "dominant": "building", "building_count": 62,
    "roof_colors": [{"name": "white", "share": 0.6, "rgb": [231, 228, 222]}, {"name": "grey", "share": 0.3, "rgb": [120, 120, 120]}],
    "confidence": 0.8, "image_size": [1024, 1024],
}


@pytest.fixture
def with_landcover(monkeypatch):
    monkeypatch.setattr(adapter, "USE_LANDCOVER", True)


def test_land_cover_is_reported_next_to_the_object_scan(image, monkeypatch, with_landcover):
    monkeypatch.setattr(adapter.land_cover_adapter, "analyze", lambda path: (LAND_FACTS, None))
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([_det("ship", 0.5, i) for i in range(3)]))

    result = adapter._handle(QueryInput(images=[image]), {})

    assert result.structured_data["land_cover"] == LAND_FACTS
    assert "building 40%" in result.text_summary and "About 62 separate building outlines" in result.text_summary
    assert "Roofs are mostly white (60%) and grey (30%)" in result.text_summary
    assert "found 3 ship(s)" in result.text_summary
    assert result.confidence == pytest.approx((0.8 + 0.5) / 2)


def test_all_three_sources_can_speak_at_once(image, monkeypatch, with_captions, with_landcover):
    monkeypatch.setattr(adapter, "_get_captioner", lambda: StubCaptioner(confidence=0.6))
    monkeypatch.setattr(adapter.land_cover_adapter, "analyze", lambda path: (LAND_FACTS, None))
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([_det("airplane", 0.5, i) for i in range(3)]))

    result = adapter._handle(QueryInput(images=[image]), {})

    assert result.structured_data["caption"] and result.structured_data["land_cover"] and result.structured_data["total_objects"] == 3
    assert result.confidence == pytest.approx((0.6 + 0.8 + 0.5) / 3)
    assert result.text_summary.index("Description") < result.text_summary.index("Land cover") < result.text_summary.index("Object scan")


def test_a_failing_land_cover_model_leaves_the_rest(image, monkeypatch, with_landcover):
    def broken(path):
        raise RuntimeError("no checkpoint")

    monkeypatch.setattr(adapter.land_cover_adapter, "analyze", broken)
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([_det("ship", 0.4, i) for i in range(3)]))

    result = adapter._handle(QueryInput(images=[image]), {})

    assert result.structured_data["land_cover"] is None and result.structured_data["total_objects"] == 3
    assert "land-cover model unavailable (no checkpoint)" in result.text_summary


def test_land_cover_alone_still_makes_a_description_when_the_scan_fails(image, monkeypatch, with_landcover):
    class BrokenScanner:
        def scan(self, *args, **kwargs):
            raise RuntimeError("groundingdino missing")

    monkeypatch.setattr(adapter.land_cover_adapter, "analyze", lambda path: (LAND_FACTS, None))
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: BrokenScanner())

    result = adapter._handle(QueryInput(images=[image]), {})

    assert result.structured_data["land_cover"] and result.confidence == pytest.approx(0.8)


def test_the_router_description_says_what_is_actually_enabled(monkeypatch):
    monkeypatch.setattr(adapter, "USE_CAPTIONS", False)
    monkeypatch.setattr(adapter, "USE_LANDCOVER", False)
    assert "does NOT describe terrain, buildings" in adapter._what_it_returns()
    monkeypatch.setattr(adapter, "USE_LANDCOVER", True)
    text = adapter._what_it_returns()
    assert "building count" in text and "roof colours" in text and "does NOT" not in text and "captioning" not in text
    monkeypatch.setattr(adapter, "USE_CAPTIONS", True)
    assert "captioning model" in adapter._what_it_returns() and "segmentation model" in adapter._what_it_returns()
