"""Pure-Python confidence combination -- no LLM involvement, so the reported confidence always
traces back to the specialist models' own outputs, never to LLM self-assessment."""

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.5


def combine_confidence(tool_confidences: list[float], warning_count: int) -> tuple[float, str]:
    """Combines one or more per-tool confidence scores into a single score + bucket.

    Multi-tool queries use the minimum (a composite answer is only as strong as its weakest leg).
    Each input-quality warning (e.g. mismatched pair dimensions) shaves a fixed penalty off the
    score, since a warning means the tools ran on data they weren't fully happy with.
    """
    if not tool_confidences:
        return 0.0, "Low"

    score = min(tool_confidences)
    score = max(0.0, score - 0.1 * warning_count)

    if score >= HIGH_THRESHOLD:
        bucket = "High"
    elif score >= MEDIUM_THRESHOLD:
        bucket = "Medium"
    else:
        bucket = "Low"
    return round(score, 4), bucket
