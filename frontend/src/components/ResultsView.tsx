import { useState } from "react";
import { reportUrl, type QueryResponse } from "../api/client";
import { CheckIcon, CopyIcon } from "./icons";
import { ExecutionTraceView } from "./ExecutionTraceView";
import { ToolResultCard } from "./ToolResultCard";

const BUCKET_CLASS: Record<string, string> = {
  High: "badge-high",
  Medium: "badge-medium",
  Low: "badge-low",
};

// What "confidence" actually means differs by tool -- some are real model scores, some are a
// data-completeness fraction, some are a generation-likelihood proxy. Showing them all with an
// identical badge invites reading them as the same kind of number, which they aren't (see each
// adapter's own docstring, e.g. groundwater_adapter.py's _data_completeness_confidence). Keyed by
// the exact ToolSpec.name values registered in backend/app/specialists/*_adapter.py.
const CONFIDENCE_SEMANTICS: Record<string, string> = {
  groundwater_potential:
    "Data completeness, not a statistical confidence: the fraction of GEE layers that had data for this location.",
  wildfire_detection:
    "NASA FIRMS' own 0-100 detection confidence for the strongest hit, rescaled to 0-1 -- not a calibrated probability.",
  visual_question_answering:
    "Mean generated-token probability -- a generation-likelihood proxy, not a calibrated chance the answer is correct.",
  change_detection:
    "Semantic model score: how far the change mask's probabilities sit from the decision boundary, averaged with the classifier's top-class probability over the changed pixels -- not a calibrated probability. (The Stage 1 pixel-difference fallback reports Otsu's between-class variance ratio instead.)",
  optical_sar_fusion:
    "Reconciliation-based: higher where the SAR and optical reads agree, lower where they disagree (surfaced, not averaged away).",
  text_guided_grounding:
    "The detector's own score for the best box. This checkpoint's scores run low even for correct boxes (real objects typically score 0.3-0.5), so a Low bucket here is not evidence the boxes are wrong -- check them against the overlay.",
  scene_description:
    "The average of what its sources reported: the captioner's mean token probability (how committed it was, not whether the description is true) and the detector's best box score (which runs low even for correct boxes). The object scan only covers airplanes, ships and storage tanks.",
  water_body_segmentation: "The segmentation model's own per-pixel confidence, averaged over the predicted mask.",
};

function confidenceTitle(result: QueryResponse): string | undefined {
  const toolName = result.execution_trace.tools_used[0]?.name;
  return toolName ? CONFIDENCE_SEMANTICS[toolName] : undefined;
}

function CopyAnswerButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access can be denied by the browser -- nothing useful to recover into, the
      // button simply doesn't confirm and the user can select-and-copy the text manually.
    }
  };

  return (
    <button className="btn btn-icon copy-btn" onClick={copy} title="Copy answer" aria-label="Copy answer">
      {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
    </button>
  );
}

export function ResultsView({ result }: { result: QueryResponse }) {
  const semantics = confidenceTitle(result);
  return (
    <div className="result">
      <div className="result-header">
        <span className={`badge ${BUCKET_CLASS[result.confidence_bucket]}`} title={semantics}>
          {result.confidence_bucket}
          <span className="badge-value">{result.confidence.toFixed(2)}</span>
          {semantics && <span className="badge-info" aria-hidden="true">ⓘ</span>}
        </span>
        <CopyAnswerButton text={result.answer_text} />
      </div>

      <p className="answer-text">{result.answer_text}</p>

      {result.tool_results.length > 0 && (
        <div className="tool-card-row">
          {result.tool_results.map((tr, i) => (
            <ToolResultCard key={`${tr.tool_name}-${i}`} result={tr} />
          ))}
        </div>
      )}

      <ExecutionTraceView trace={result.execution_trace} />

      <a className="download-link" href={reportUrl(result.query_id)} target="_blank" rel="noreferrer">
        Download report
      </a>
    </div>
  );
}
