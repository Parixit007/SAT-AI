"""The parts of models/grounding/grounding_tool.py that don't need the model or the vendored checkout."""

import numpy as np
import pytest
from PIL import Image

from app.config import MODELS_DIR
from app.specialists._loader import load_module

tool = load_module(MODELS_DIR / "grounding" / "grounding_tool.py", "grounding_tool_pure")


@pytest.mark.parametrize("size, expected", [
    # (width, height) -> (height, width); values checked against the checkout's own transform
    ((1024, 822), (800, 996)),
    ((4000, 3000), (800, 1066)),
    ((512, 512), (800, 800)),
    ((300, 1500), (1335, 267)),
    ((1333, 800), (800, 1333)),
])
def test_resize_target_matches_the_training_transform(size, expected):
    assert tool._resize_target(*size) == expected


def test_preprocess_produces_a_normalised_chw_tensor():
    image = Image.new("RGB", (1024, 822), (124, 116, 104))  # ~ the ImageNet mean colour
    tensor = tool._preprocess(image)
    assert tuple(tensor.shape) == (3, 800, 996)
    assert abs(float(tensor.mean())) < 0.05


@pytest.mark.parametrize("n", [1, 5, 40])
def test_draw_boxes_handles_a_single_match_and_a_crowd(tmp_path, n):
    source = tmp_path / "in.png"
    Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8)).save(source)
    detections = [{"phrase": "airplane" if i % 3 else "ship", "bbox_xyxy": [i, i, i + 30.0, i + 20.0], "score": 0.4} for i in range(n)]
    out = tmp_path / "out.jpg"
    tool.draw_boxes(str(source), detections, str(out))
    assert Image.open(out).size == (300, 200)
    assert np.asarray(Image.open(out)).max() > 0  # something was drawn on the black image


class _NoModelTool(tool.GroundingTool):
    """GroundingTool with the network replaced by fixed outputs, so ground()'s own logic can be tested."""

    def __init__(self, logits, boxes):  # deliberately skips the real __init__ (no checkpoint, no checkout)
        import types
        self.model = types.SimpleNamespace(tokenizer=lambda caption: {})
        self._fixed = (logits, boxes)

    def _forward(self, image, caption):
        return self._fixed

    def _get_phrases(self, mask, tokenized, tokenizer):
        return "car"


def _image(tmp_path):
    path = tmp_path / "scene.png"
    Image.new("RGB", (100, 100)).save(path)
    return str(path)


def test_a_query_with_no_match_returns_an_empty_list_instead_of_raising(tmp_path):
    import torch

    quiet = _NoModelTool(torch.zeros((5, 8)), torch.rand((5, 4)))  # every logit below the 0.25 threshold
    assert quiet.ground(_image(tmp_path), "airplane") == []


def test_matches_come_back_in_pixel_coordinates_best_first(tmp_path):
    import torch

    logits = torch.zeros((3, 8))
    logits[0, 0], logits[1, 0] = 0.40, 0.70  # two boxes clear the threshold, the third does not
    boxes = torch.tensor([[0.50, 0.50, 0.20, 0.20], [0.20, 0.20, 0.10, 0.10], [0.90, 0.90, 0.10, 0.10]])
    found = _NoModelTool(logits, boxes).ground(_image(tmp_path), "car")
    assert [d["score"] for d in found] == [0.7, 0.4]
    assert found[0]["bbox_xyxy"] == [15.0, 15.0, 25.0, 25.0]
    assert found[1]["bbox_xyxy"] == [40.0, 40.0, 60.0, 60.0]
    assert {d["phrase"] for d in found} == {"car"}
