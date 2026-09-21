"""models/landcover/landcover_tool.py without a checkpoint: the pure functions that turn a class map into the
numbers the app reports (area fractions, building count, roof colours), the tiling arithmetic, and the
whole-image / tiled prediction and analyze() paths run through a stub network."""

import numpy as np
import pytest
from PIL import Image

from app.config import MODELS_DIR
from app.specialists._loader import load_module

lc = load_module(MODELS_DIR / "landcover" / "landcover_tool.py", "landcover_tool_under_test")
BUILDING, TREE = lc.CLASSES.index("building"), lc.CLASSES.index("tree")


def test_class_fractions_are_shares_of_all_pixels():
    class_map = np.array([[BUILDING, BUILDING, TREE], [TREE, TREE, TREE]], dtype=np.uint8)
    fractions = lc.class_fractions(class_map)
    assert fractions["building"] == pytest.approx(2 / 6) and fractions["tree"] == pytest.approx(4 / 6)
    assert fractions["water"] == 0.0 and sum(fractions.values()) == pytest.approx(1.0)


def test_a_building_count_ignores_specks_and_joins_diagonal_neighbours():
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:8, 2:8] = True            # 36 px: a building
    mask[20:26, 20:26] = True        # 36 px: a second one
    mask[30:32, 30:32] = True        # 4 px: a speck
    mask[10, 30], mask[11, 31] = True, True  # 2 px touching diagonally: still too small
    assert lc.count_components(mask) == 2
    diagonal = np.zeros((20, 20), dtype=bool)
    for i in range(8):
        diagonal[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = True  # 2x2 blocks chained corner to corner: one 32 px region
    assert lc.count_components(diagonal) == 1
    assert lc.count_components(np.zeros((10, 10), dtype=bool)) == 0


@pytest.mark.parametrize("rgb, expected", [
    ((255, 255, 255), "white"), ((200, 200, 200), "light grey"), ((128, 128, 128), "grey"),
    ((70, 70, 70), "dark grey"), ((10, 10, 10), "very dark"),
    ((200, 40, 40), "red"), ((120, 60, 30), "brown"), ((230, 140, 40), "orange"), ((200, 170, 130), "tan"),
    ((230, 220, 60), "yellow"), ((60, 160, 60), "green"), ((50, 90, 200), "blue"), ((150, 60, 170), "purple"),
])
def test_colours_get_their_plain_names(rgb, expected):
    assert lc.name_color(rgb) == expected


def test_roof_colours_come_from_inside_the_building_mask_only():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:] = (50, 90, 200)                                  # blue everywhere...
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:50], mask[10:90, 50:90] = True, True       # ...except a building half white, half red
    image[10:90, 10:50] = (240, 240, 240)
    image[10:90, 50:90] = (200, 40, 40)
    colours = lc.roof_colors(image, mask)
    assert {c["name"] for c in colours} == {"white", "red"}     # the blue outside the mask never votes
    assert all(0.45 <= c["share"] <= 0.55 for c in colours)
    white = next(c for c in colours if c["name"] == "white")
    assert white["rgb"] == [240, 240, 240]


def test_too_few_building_pixels_give_no_roof_colours():
    image = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:9, 5:9] = True
    assert lc.roof_colors(image, mask) == []


@pytest.mark.parametrize("length, tile, overlap", [(1000, 768, 128), (2000, 768, 128), (4000, 768, 128), (768, 768, 128), (500, 768, 128)])
def test_tiles_cover_the_whole_length_and_stay_inside_it(length, tile, overlap):
    starts = lc.tile_starts(length, tile, overlap)
    covered = np.zeros(length, dtype=bool)
    for s in starts:
        assert 0 <= s and (s + tile <= length or length <= tile)
        covered[s:s + tile] = True
    assert covered.all()
    for a, b in zip(starts, starts[1:]):
        assert a + tile - b >= overlap  # neighbouring tiles share at least `overlap` pixels


class _AlwaysBuilding:
    """A stand-in network: every pixel is a building, whatever the input (checks shapes and stitching, not accuracy)."""

    def __init__(self, torch):
        self.torch = torch

    def __call__(self, x):
        logits = self.torch.zeros(x.shape[0], len(lc.CLASSES), *x.shape[-2:])
        logits[:, BUILDING] = 5.0
        return logits


def _stub_tool():
    import torch

    tool = object.__new__(lc.LandCoverTool)
    tool._torch, tool.device, tool.model = torch, "cpu", _AlwaysBuilding(torch)
    tool._mean = torch.zeros(3, 1, 1)
    tool._std = torch.ones(3, 1, 1)
    return tool


@pytest.mark.parametrize("size", [(260, 300), (2000, 1700)])  # a small odd-sized image, and one large enough to be tiled
def test_prediction_returns_full_size_probabilities_for_small_and_tiled_images(size):
    w, h = size
    probabilities = _stub_tool()._probabilities(np.zeros((h, w, 3), dtype=np.uint8))
    assert probabilities.shape == (len(lc.CLASSES), h, w)
    assert np.allclose(probabilities.sum(0), 1.0, atol=1e-4)
    assert (probabilities.argmax(0) == BUILDING).all()


def test_analyze_reports_fractions_count_colours_and_confidence(tmp_path):
    path = tmp_path / "roofs.png"
    Image.fromarray(np.full((120, 160, 3), (230, 228, 222), dtype=np.uint8)).save(path)
    result = _stub_tool().analyze(str(path))
    assert result["class_map"].shape == (120, 160) and result["image_size"] == [160, 120]
    assert result["fractions"]["building"] == pytest.approx(1.0)
    assert result["building_count"] == 1                   # one connected region: the whole frame
    assert [c["name"] for c in result["roof_colors"]] == ["white"]
    assert 0.9 < result["confidence"] <= 1.0


def test_the_overlay_adds_a_legend_strip_below_the_image(tmp_path):
    src = tmp_path / "scene.png"
    Image.fromarray(np.zeros((80, 120, 3), dtype=np.uint8)).save(src)
    class_map = np.full((80, 120), TREE, dtype=np.uint8)
    class_map[:, 60:] = BUILDING
    out = tmp_path / "overlay.jpg"
    lc.draw_overlay(str(src), class_map, str(out))
    assert Image.open(out).size == (120, 80 + 22)
