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


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "scene.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    return path


def _det(label, score, i=0):
    return {"phrase": label, "bbox_xyxy": [i, i, i + 8.0, i + 8.0], "score": score}


def test_it_is_registered_as_a_single_image_tool():
    spec = DEFAULT_REGISTRY.get("scene_description")
    assert spec is not None and (spec.min_images, spec.max_images) == (1, 1) and not spec.requires_location


def test_it_counts_each_category_and_says_what_it_does_not_check(image, monkeypatch):
    scanner = StubScanner([_det("airplane", 0.5, i) for i in range(3)] + [_det("ship", 0.35), _det("airplane", 0.31, 9)])
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: scanner)

    result = adapter._handle(QueryInput(images=[image]), {})

    assert scanner.calls == [{"categories": ["airplane", "ship", "storage tank"], "box_threshold": 0.30}]
    assert result.structured_data["objects"] == [
        {"label": "airplane", "count": 4, "best_score": 0.5},
        {"label": "ship", "count": 1, "best_score": 0.35},
    ]
    assert result.structured_data["total_objects"] == 5
    assert "checks only airplane, ship and storage tank" in result.text_summary
    assert "4 airplane(s); 1 ship(s)" in result.text_summary
    assert result.confidence == 0.5
    assert result.evidence_image_path is not None and result.evidence_image_path.exists()


def test_an_empty_scan_says_so_and_has_zero_confidence(image, monkeypatch):
    monkeypatch.setattr(grounding_adapter, "_get_tool", lambda: StubScanner([]))
    result = adapter._handle(QueryInput(images=[image]), {})
    assert "none found" in result.text_summary
    assert result.confidence == 0.0 and result.evidence_image_path is None and result.structured_data["objects"] == []
