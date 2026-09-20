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
