from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.store import get_query

router = APIRouter()


@router.get("/report/{query_id}")
async def get_report(query_id: str) -> Response:
    """Plain-text downloadable report bundling a past query's answer, confidence, and execution
    trace. A polished PDF/HTML export is future work -- this satisfies the spec's "downloadable
    report" requirement at MVP fidelity."""
    result = get_query(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown query_id '{query_id}'.")

    trace = result.trace
    lines = [
        "SatQuery AI -- Execution Report",
        f"Generated: {trace.timestamp}",
        "",
        f"Answer: {result.answer_text}",
        f"Confidence: {result.confidence} ({result.confidence_bucket})",
        "",
        f"Selected task: {trace.selected_task}",
        "Tools used:",
    ]
    for t in trace.tools_used:
        lines.append(f"  - [round {t.get('round', 1)}] {t['name']} (checkpoint: {t.get('checkpoint_id')}, params: {t['params']})")
    if trace.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in trace.warnings:
            lines.append(f"  - {w}")
    lines.append("")
    lines.append("Input summary:")
    lines.append(trace.input_summary)

    return Response(
        content="\n".join(lines),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="satquery_report_{query_id}.txt"'},
    )
