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
