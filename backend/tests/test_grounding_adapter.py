"""The grounding adapter's count path and its query normalisation. The detector is stubbed; what is
pinned here is what the adapter does with what it returns."""

import numpy as np
import pytest
from PIL import Image

from app.orchestrator.tool_registry import QueryInput
from app.specialists import grounding_adapter as adapter


@pytest.mark.parametrize("raw, expected", [
    ("airplane", "airplane"),
    ("aeroplanes", "airplane"),                                   # British plural: 19 boxes vs 39 on a real scene
    ("how many aeroplanes do you see", "airplane"),               # the whole question: 3 boxes on a real scene
    ("How many airplanes are in the image?", "airplane"),
    ("how many storage tanks are there in this image", "storage tank"),
    ("is there a stadium in the image?", "stadium"),
    ("Are there any planes visible?", "airplane"),
    ("count the number of ships", "ship"),
    ("find all the vehicles", "vehicle"),
    ("show me the harbour", "harbor"),
    ("ships. vehicles", "ship . vehicle"),
    ("ship . vehicle . airplane .", "ship . vehicle . airplane"),
    ("the ship near the harbor entrance", "the ship near the harbor entrance"),                   # a real referring expression is left alone
    ("the large white airplane on the left apron", "the large white airplane on the left apron"),
])
def test_normalize_category_query(raw, expected):
    assert adapter.normalize_category_query(raw) == expected


class StubDetector:
    def __init__(self, detections):
        self.detections, self.calls = detections, []

    def ground(self, image_path, query, top_k=None, **kwargs):
        self.calls.append({"query": query, "top_k": top_k})
        return self.detections if top_k is None else self.detections[:top_k]


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "scene.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    return path


def _detections(n, score=0.4):
    return [{"phrase": "airplane", "bbox_xyxy": [i, i, i + 8.0, i + 8.0], "score": score - i / 1000} for i in range(n)]


def test_counts_every_detection_and_reports_the_count(image, monkeypatch):
    detector = StubDetector(_detections(39))
    monkeypatch.setattr(adapter, "_get_tool", lambda: detector)

    result = adapter._handle(QueryInput(images=[image]), {"query": "aeroplanes"})

    assert detector.calls == [{"query": "airplane", "top_k": None}]  # normalised, and no cap on the matches
    assert result.structured_data["count"] == 39 and result.structured_data["by_phrase"] == {"airplane": 39}
    assert result.structured_data["original_query"] == "aeroplanes"
    assert "Found 39 region(s) matching 'airplane'" in result.text_summary
    assert "Best match" not in result.text_summary  # one box's coordinates say nothing about a count of 39
    assert result.evidence_image_path is not None and result.evidence_image_path.exists()
    assert result.confidence == pytest.approx(0.4)


def test_an_explicit_top_k_is_still_honoured(image, monkeypatch):
    detector = StubDetector(_detections(10))
    monkeypatch.setattr(adapter, "_get_tool", lambda: detector)
    result = adapter._handle(QueryInput(images=[image]), {"query": "airplane", "top_k": 3})
    assert result.structured_data["count"] == 3


def test_a_single_match_reports_where_it_is(image, monkeypatch):
    monkeypatch.setattr(adapter, "_get_tool", lambda: StubDetector(_detections(1)))
    result = adapter._handle(QueryInput(images=[image]), {"query": "the ship near the harbor entrance"})
    assert "Best match" in result.text_summary
    assert "original_query" not in result.structured_data  # nothing was normalised


def test_no_matches(image, monkeypatch):
    monkeypatch.setattr(adapter, "_get_tool", lambda: StubDetector([]))
    result = adapter._handle(QueryInput(images=[image]), {"query": "airplane"})
    assert result.structured_data["count"] == 0 and result.confidence == 0.0
    assert result.evidence_image_path is None
    assert "No regions matching 'airplane'" in result.text_summary


# ---------------------------------------------------------------- tiled counting of small objects

def _box(x1, y1, x2, y2, score=0.4, phrase="vehicle"):
    return {"phrase": phrase, "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)], "score": score}


class TileDetector:
    """Finds one 'vehicle' in the middle of every tile it is shown, and remembers the tile sizes it saw."""

    def __init__(self):
        self.tile_sizes, self.queries = [], []

    def ground(self, image_path, query, top_k=None, **kwargs):
        with Image.open(image_path) as tile:
            self.tile_sizes.append(tile.size)
            w, h = tile.size
        self.queries.append(query)
        return [_box(w / 2 - 10, h / 2 - 5, w / 2 + 10, h / 2 + 5)]


@pytest.fixture
def street(tmp_path):
    path = tmp_path / "street.png"
    Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8)).save(path)
    return path


@pytest.mark.parametrize("length, expected", [(300, [0]), (512, [0]), (1024, [0, 352, 512]), (2000, [0, 352, 704, 1056, 1408, 1488])])
def test_tile_starts_cover_the_frame_with_the_stated_overlap(length, expected):
    starts = adapter.tile_starts(length, adapter.TILE, adapter.TILE_OVERLAP)
    assert starts == expected
    assert starts[-1] + min(adapter.TILE, length) == length


def test_the_same_object_seen_in_two_tiles_is_counted_once():
    merged = adapter.merge_boxes([_box(100, 100, 120, 110, 0.5), _box(101, 100, 121, 110, 0.4), _box(300, 300, 320, 310, 0.3)])
    assert [d["score"] for d in merged] == [0.5, 0.3]


def test_a_vehicle_question_on_a_large_frame_is_counted_tile_by_tile(street, monkeypatch):
    detector = TileDetector()
    monkeypatch.setattr(adapter, "_get_tool", lambda: detector)
    monkeypatch.setattr(adapter, "EVIDENCE_DIR", street.parent)

    result = adapter._handle(QueryInput(images=[street]), {"query": "how many cars are in this image"})

    assert set(detector.tile_sizes) == {(512, 512)} and len(detector.tile_sizes) == 9  # 3 x 3 tiles at native resolution
    assert set(detector.queries) == {"cars"}                                     # the wording the detector responds to
    assert {d["phrase"] for d in result.structured_data["detections"]} == {"vehicle"}  # labelled as the user asked
    assert result.structured_data["query"] == "vehicle"
    assert result.structured_data["tiles"] == 9 and result.structured_data["count"] == 9  # one per tile, distinct positions
    assert "lower bound" in result.text_summary and "9 overlapping full-resolution tiles" in result.text_summary
    boxes = [d["bbox_xyxy"] for d in result.structured_data["detections"]]
    assert [246.0, 251.0, 266.0, 261.0] in boxes  # the first tile starts at (0, 0): its box is unshifted
    assert [758.0, 763.0, 778.0, 773.0] in boxes  # the last tile starts at (512, 512): its box is offset by that


def test_other_queries_and_small_frames_are_not_tiled(street, image, monkeypatch):
    detector = TileDetector()
    monkeypatch.setattr(adapter, "_get_tool", lambda: detector)
    adapter._handle(QueryInput(images=[street]), {"query": "airplane"})        # a large frame, but not a small-object query
    assert len(detector.tile_sizes) == 1 and detector.tile_sizes[0] == (1024, 1024)
    detector.tile_sizes.clear()
    result = adapter._handle(QueryInput(images=[image]), {"query": "vehicle"})  # a small-object query, but a 64 px frame
    assert detector.tile_sizes == [(64, 64)] and detector.queries[-1] == "cars"  # still asked the better way, just not tiled
    assert "tiles" not in result.structured_data
