"""The pure parts of models/captioning/caption_tool.py -- the model itself needs a ~1GB checkpoint, so
what is testable everywhere is how a checkpoint's metadata maps to writing styles (round one's
single-prompt checkpoint must keep working next to a two-mode one) and the sentence trimming."""

import pytest

from app.config import MODELS_DIR
from app.specialists._loader import load_module

caption_tool = load_module(MODELS_DIR / "captioning" / "caption_tool.py", "caption_tool_under_test")


def test_a_round_one_checkpoint_is_the_detailed_style_only():
    meta = {"prompt": "Describe the image in detail.", "fine_tuned_metrics": {}}
    assert caption_tool.prompts_from_meta(meta) == {"detailed": "Describe the image in detail."}


def test_a_two_mode_checkpoint_lists_both_styles():
    meta = {
        "prompt": "Describe the image in detail.",  # kept for older readers, must not shadow "prompts"
        "prompts": {"detailed": "Describe the image in detail.", "brief": "Briefly describe the scene."},
    }
    assert caption_tool.prompts_from_meta(meta) == {
        "detailed": "Describe the image in detail.",
        "brief": "Briefly describe the scene.",
    }


def test_the_returned_prompts_are_a_copy_not_the_metadata_itself():
    meta = {"prompt": "x", "prompts": {"detailed": "x"}}
    caption_tool.prompts_from_meta(meta)["brief"] = "y"
    assert "brief" not in meta["prompts"]


def test_every_style_has_a_generation_budget():
    assert set(caption_tool.MAX_NEW_TOKENS) >= {"detailed", "brief"}
    assert caption_tool.MAX_NEW_TOKENS["brief"] < caption_tool.MAX_NEW_TOKENS["detailed"]


@pytest.mark.parametrize("text, expected", [
    ("A whole sentence.", "A whole sentence."),
    ("First one. Second one cut off mid", "First one."),
    ("Only a fragment with no full stop", "Only a fragment with no full stop"),
    ("", ""),
])
def test_a_caption_cut_off_by_the_token_limit_keeps_only_complete_sentences(text, expected):
    assert caption_tool.trim_to_sentence(text) == expected
