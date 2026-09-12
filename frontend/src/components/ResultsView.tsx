import { evidenceImageUrl, reportUrl, type QueryResponse } from "../api/client";
import { ExecutionTraceView } from "./ExecutionTraceView";

const BUCKET_CLASS: Record<string, string> = {
  High: "badge-high",
  Medium: "badge-medium",
  Low: "badge-low",
};

export function ResultsView({ result }: { result: QueryResponse }) {
  return (
    <div className="result">
      <div className="result-header">
        <span className={`badge ${BUCKET_CLASS[result.confidence_bucket]}`}>
          {result.confidence_bucket}
          <span className="badge-value">{result.confidence.toFixed(2)}</span>
        </span>
      </div>

      <p className="answer-text">{result.answer_text}</p>

      {result.evidence_image_urls.length > 0 && (
        <div className="evidence-row">
          {result.evidence_image_urls.map((url) => (
            <img key={url} src={evidenceImageUrl(url)} alt="Visual evidence" className="evidence-image" />
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
