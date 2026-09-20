"""Tests for the Stage 2 semantic change specialist (models/change_detection/semantic_change_tool.py
+ the change_detection adapter's Stage 2 path). No trained checkpoint is needed or used: the pure
logic is tested directly, and the model plumbing runs against a randomly-initialised network saved
in exactly the format the training notebook exports -- these check that everything is wired
correctly, not that the model is accurate (that number comes from the Kaggle run)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import app.specialists.change_detection_adapter as adapter
from app.config import MODELS_DIR, REPO_ROOT
from app.orchestrator.tool_registry import QueryInput
from app.specialists._loader import load_module

TOOL_PATH = MODELS_DIR / "change_detection" / "semantic_change_tool.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "kaggle_finetune_change_segmentation_second.ipynb"


@pytest.fixture(scope="module")
def sct():
    return load_module(TOOL_PATH, "semantic_change_tool")


# --- the notebook and the tool must share one model definition ------------------------------------


def _model_block(source: str) -> str:
    begin = source.index("# --- BEGIN SHARED MODEL BLOCK")
    end_marker = source.index("# --- END SHARED MODEL BLOCK")
    end = source.find("\n", end_marker)  # the block is the last thing in the notebook's cell: no newline after it
    return source[begin : end if end != -1 else len(source)].rstrip("\n")


def test_notebook_model_block_is_identical_to_the_tool_copy():
    """The training notebook can't import from the repo, so it carries a copy of the model
    definition. If the two drift, a checkpoint trained by the notebook would fail to load in the
    tool (strict=True), or worse, load into a subtly different architecture -- caught here instead."""
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    matching = [c for c in cells if "BEGIN SHARED MODEL BLOCK" in c]
    assert len(matching) == 1, "expected exactly one notebook cell carrying the shared model block"
    assert _model_block(matching[0]) == _model_block(TOOL_PATH.read_text())


# --- compute_class_changes: pure numpy ------------------------------------------------------------

NAMES = ["water", "bare ground", "low vegetation", "trees", "buildings", "playground"]


def test_class_changes_known_case(sct):
    # 4x4 scene: 4 changed pixels. Two go bare ground -> buildings, one low vegetation -> buildings,
    # one trees -> bare ground.
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :] = True
    before = np.zeros((4, 4), dtype=np.uint8)
    after = np.zeros((4, 4), dtype=np.uint8)
    before[0] = [1, 1, 2, 3]
    after[0] = [4, 4, 4, 1]

    out = sct.compute_class_changes(mask, before, after, NAMES, tolerance=0.005)

    assert out["change_fraction"] == pytest.approx(4 / 16)
    by_name = {c["class"]: c for c in out["class_changes"]}
    assert by_name["buildings"]["before"] == 0 and by_name["buildings"]["after"] == pytest.approx(3 / 16, abs=1e-4)
    assert by_name["buildings"]["net"] == pytest.approx(3 / 16, abs=1e-4)
    assert by_name["buildings"]["direction"] == "increased"
    assert by_name["low vegetation"]["direction"] == "decreased"
    assert by_name["trees"]["direction"] == "decreased"
    assert by_name["water"]["direction"] == "unchanged"
    # bare ground: lost 2 (-> buildings), gained 1 (from trees) => net -1/16
    assert by_name["bare ground"]["net"] == pytest.approx(-1 / 16, abs=1e-4)
    top = out["transitions"][0]
    assert (top["from"], top["to"]) == ("bare ground", "buildings") and top["fraction"] == pytest.approx(2 / 16, abs=1e-4)


def test_class_changes_net_sums_to_zero_and_before_after_sum_to_change_fraction(sct):
    """Every changed pixel adds one to 'after' and one to 'before', so net change over all classes
    must cancel -- the invariant that makes per-class net change well-defined from changed pixels
    alone (unchanged pixels cancel out of area(t2) - area(t1))."""
    rng = np.random.default_rng(0)
    mask = rng.random((64, 64)) > 0.6
    before = rng.integers(0, 6, (64, 64)).astype(np.uint8)
    after = rng.integers(0, 6, (64, 64)).astype(np.uint8)

    out = sct.compute_class_changes(mask, before, after, NAMES)

    assert sum(c["net"] for c in out["class_changes"]) == pytest.approx(0, abs=1e-3)  # 6 values rounded to 4dp
    assert sum(c["before"] for c in out["class_changes"]) == pytest.approx(out["change_fraction"], abs=1e-3)
    assert sum(c["after"] for c in out["class_changes"]) == pytest.approx(out["change_fraction"], abs=1e-3)


def test_class_changes_ignores_prediction_values_outside_the_mask(sct):
    """Class maps carry an ignore value (255) at unchanged pixels -- it must never be read."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[2, 2] = True
    before = np.full((8, 8), 255, dtype=np.uint8)
    after = np.full((8, 8), 255, dtype=np.uint8)
    before[2, 2], after[2, 2] = 1, 4

    out = sct.compute_class_changes(mask, before, after, NAMES)

    assert out["change_fraction"] == pytest.approx(1 / 64, abs=1e-4)


def test_class_changes_with_no_change_is_all_zero(sct):
    mask = np.zeros((8, 8), dtype=bool)
    z = np.zeros((8, 8), dtype=np.uint8)
    out = sct.compute_class_changes(mask, z, z, NAMES)
    assert out["change_fraction"] == 0
    assert out["transitions"] == []
    assert all(c["direction"] == "unchanged" for c in out["class_changes"])


# --- describe_changes ------------------------------------------------------------------------------


def _summary(sct, before_cls, after_cls, frac_pixels):
    mask = np.zeros((100, 100), dtype=bool)
    mask.ravel()[:frac_pixels] = True
    b = np.full((100, 100), before_cls, dtype=np.uint8)
    a = np.full((100, 100), after_cls, dtype=np.uint8)
    out = sct.compute_class_changes(mask, b, a, NAMES)
    return sct.describe_changes(out)


def test_describe_always_states_the_building_direction(sct):
    """'Has the built-up area increased, decreased, or remained unchanged?' is one of the spec's own
    representative queries -- silence about buildings would read as no answer rather than 'no change'."""
    up = _summary(sct, 2, 4, 1500)  # low vegetation -> buildings
    down = _summary(sct, 4, 2, 1500)  # buildings -> low vegetation
    flat = _summary(sct, 1, 2, 1500)  # bare ground -> low vegetation, buildings untouched

    assert "Built-up area (buildings) increased by 15.0% of the scene" in up
    assert "Built-up area (buildings) decreased by 15.0% of the scene" in down
    assert "Built-up area (buildings) is essentially unchanged" in flat
    assert "low vegetation to buildings (15.0%)" in up


def test_describe_no_change(sct):
    mask = np.zeros((10, 10), dtype=bool)
    z = np.zeros((10, 10), dtype=np.uint8)
    text = sct.describe_changes(sct.compute_class_changes(mask, z, z, NAMES))
    assert "No land-cover change" in text


# --- the model + tool wiring (random weights, notebook's checkpoint format) -----------------------


@pytest.fixture(scope="module")
def random_checkpoint(sct, tmp_path_factory):
    torch.manual_seed(0)
    model = sct.SiameseSCDNet(encoder_name="resnet34", encoder_weights=None)
    path = tmp_path_factory.mktemp("ckpt") / "semantic_change_unet.pt"
    # Exactly the shape the notebook's save_checkpoint() writes: tensors + plain python builtins.
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "encoder_name": "resnet34",
            "img_size": 64,  # small so the test is fast; the tool resizes to whatever the checkpoint says
            "class_names": list(sct.CLASS_NAMES),
            "change_threshold": 0.4,
            "metrics": {"val_Score": 0.1, "test_net_MAE_pct": {"buildings": 1.5}, "epochs": 2},
        },
        path,
    )
    return path


def test_model_output_shapes(sct):
    model = sct.SiameseSCDNet(encoder_weights=None).eval()
    a = torch.randn(2, 3, 64, 64)
    change, sem_a, sem_b = model(a, torch.randn(2, 3, 64, 64))
    assert change.shape == (2, 1, 64, 64)
    assert sem_a.shape == sem_b.shape == (2, 6, 64, 64)


def _write_pair(tmp_path, size=(96, 80)):
    rng = np.random.default_rng(1)
    paths = []
    for name in ("before.png", "after.png"):
        p = tmp_path / name
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def test_tool_loads_exported_checkpoint_and_analyzes(sct, random_checkpoint, tmp_path):
    before, after = _write_pair(tmp_path)  # deliberately not square, not the training size
    tool = sct.SemanticChangeTool(str(random_checkpoint), device="cpu")

    assert tool.change_threshold == pytest.approx(0.4)  # picked up from the checkpoint
    result = tool.analyze(str(before), str(after), change_threshold=0.0)  # force everything "changed"

    assert result["change_mask"].shape == (80, 96)  # back at the input's own resolution
    assert result["change_fraction"] == 1.0
    assert result["largest_region_bbox"] == [0, 0, 96, 80]
    assert len(result["class_changes"]) == 6
    assert sum(c["net"] for c in result["class_changes"]) == pytest.approx(0, abs=1e-3)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["sem_before"].shape == (80, 96)

    out = tmp_path / "overlay.jpg"
    sct.draw_semantic_change_overlay(str(before), str(after), result, str(out))
    composite = Image.open(out)
    assert composite.width == 96 * 3 and composite.height > 80  # three panels + a legend strip


def test_unchanged_pixels_carry_the_ignore_value(sct, random_checkpoint, tmp_path):
    before, after = _write_pair(tmp_path)
    tool = sct.SemanticChangeTool(str(random_checkpoint), device="cpu")
    result = tool.analyze(str(before), str(after), change_threshold=1.01)  # nothing can exceed this

    assert result["change_fraction"] == 0
    assert (result["sem_before"] == sct.IGNORE_INDEX).all()
    assert result["largest_region_bbox"] is None


# --- the adapter's Stage 2 path --------------------------------------------------------------------


def test_adapter_stage2_result_shape(sct, random_checkpoint, tmp_path, monkeypatch):
    before, after = _write_pair(tmp_path)
    monkeypatch.setattr(adapter, "USE_STAGE2", True)
    monkeypatch.setattr(adapter, "CHANGE_SEG_CHECKPOINT", random_checkpoint)
    monkeypatch.setattr(adapter, "_stage2_tool", None)  # don't reuse a real one if some other test built it

    result = adapter._handle(QueryInput(images=[before, after]), {})

    assert result.tool_name == "change_detection"
    data = result.structured_data
    assert data["method"] == "semantic"
    assert {"change_fraction", "largest_region_bbox", "class_changes", "transitions"} <= set(data)
    assert len(data["class_changes"]) == 6
    # the fields Stage 1 also returns must keep their names/types -- the UI's Stage 1 rendering reads them
    assert isinstance(data["change_fraction"], float)
    assert "Built-up area (buildings)" in result.text_summary or "No land-cover change" in result.text_summary
    assert result.evidence_image_path is not None and result.evidence_image_path.exists()
    assert 0.0 <= result.confidence <= 1.0
    result.evidence_image_path.unlink()


def test_adapter_stage1_marks_its_method(change_pair_images, monkeypatch):
    before, after, _ = change_pair_images
    monkeypatch.setattr(adapter, "USE_STAGE2", False)

    result = adapter._handle(QueryInput(images=[before, after]), {})

    assert result.structured_data["method"] == "pixel_difference"
    assert "class_changes" not in result.structured_data
    result.evidence_image_path.unlink()


def test_adapter_description_matches_the_active_stage():
    """The description is what the LLM router sees -- it must not promise class-level answers while
    Stage 1 (which can't give them) is what actually runs."""
    if adapter.USE_STAGE2:
        assert "WHICH land-cover classes" in adapter.TOOL_SPEC.description
        assert adapter.TOOL_SPEC.checkpoint_id == adapter.CHANGE_SEG_CHECKPOINT.name
    else:
        assert "not which kind of land cover changed" in adapter.TOOL_SPEC.description
        assert adapter.TOOL_SPEC.checkpoint_id is None
