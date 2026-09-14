"""Direct unit tests for combine_confidence() -- previously only exercised indirectly through
orchestrator tests that each fed it at most one tool result and zero-or-incidental warnings, so
the min-of-multiple and the warning-penalty arithmetic were never pinned to an exact expected
number anywhere."""

import pytest

from app.orchestrator.confidence import combine_confidence


def test_no_tool_results_is_zero_confidence_low():
    assert combine_confidence([], warning_count=0) == (0.0, "Low")


def test_multi_tool_confidence_is_the_minimum_not_average():
    # A composite answer is only as strong as its weakest leg -- 0.4 here, not (0.9+0.4)/2.
    score, bucket = combine_confidence([0.9, 0.4], warning_count=0)
    assert score == 0.4
    assert bucket == "Low"


def test_each_warning_subtracts_a_flat_tenth():
    score, bucket = combine_confidence([0.9], warning_count=2)
    assert score == pytest.approx(0.7)
    assert bucket == "Medium"


def test_penalty_floors_at_zero_not_negative():
    score, bucket = combine_confidence([0.05], warning_count=5)
    assert score == 0.0
    assert bucket == "Low"


@pytest.mark.parametrize(
    "raw_score,expected_bucket",
    [
        (0.75, "High"),  # exactly at HIGH_THRESHOLD -- >=, so this side counts as High
        (0.7499, "Medium"),
        (0.5, "Medium"),  # exactly at MEDIUM_THRESHOLD -- >=, so this side counts as Medium
        (0.4999, "Low"),
    ],
)
def test_bucket_boundaries(raw_score, expected_bucket):
    _, bucket = combine_confidence([raw_score], warning_count=0)
    assert bucket == expected_bucket


def test_score_is_rounded_to_four_places():
    score, _ = combine_confidence([1 / 3], warning_count=0)
    assert score == 0.3333
