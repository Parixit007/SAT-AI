import { motion } from "framer-motion";
import type { QueryResponse } from "../api/client";
import { TOOL_ICON } from "../toolMeta";
import { RetryIcon } from "./icons";
import { ResultsView } from "./ResultsView";

// One query -> one result, read top-to-bottom as a report entry, not a left/right chat exchange --
// see the UI-overhaul plan's "tool, not chatbot" amendment. A single entry carries its own
// queryText through every status so an errored entry can retry without the caller re-threading it.
export type QueryEntryState =
  | { id: string; queryText: string; status: "pending" }
  | { id: string; queryText: string; status: "done"; result: QueryResponse }
  | { id: string; queryText: string; status: "error"; error: string };

function entryIcon(entry: QueryEntryState): string | null {
  if (entry.status !== "done") return null;
  const name = entry.result.execution_trace.tools_used[0]?.name;
  return name ? TOOL_ICON[name] ?? null : null;
}

export function QueryEntry({ entry, onRetry }: { entry: QueryEntryState; onRetry: (queryText: string) => void }) {
  const icon = entryIcon(entry);

  return (
    <motion.div
      className="entry"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.26, ease: [0.2, 0.8, 0.2, 1] }}
    >
      <div className="entry-query">
        <span className="entry-query-label">Query</span>
        <span className="entry-query-text">{entry.queryText}</span>
        {icon && <span className="entry-tool-icon" aria-hidden="true">{icon}</span>}
      </div>

      <div className="entry-result panel">
        {entry.status === "pending" && (
          <div className="entry-pending">
            <span className="pulse-bar" aria-hidden="true">
              <span />
            </span>
            Processing…
          </div>
        )}

        {entry.status === "error" && (
          <div className="entry-error">
            <span>{entry.error}</span>
            <button className="btn btn-text retry-btn" onClick={() => onRetry(entry.queryText)}>
              <RetryIcon size={13} /> Retry
            </button>
          </div>
        )}

        {entry.status === "done" && <ResultsView result={entry.result} />}
      </div>
    </motion.div>
  );
}
