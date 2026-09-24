import { useState } from "react";
import type { ExecutionTrace, ToolUsage } from "../api/client";

// One <li> for a single tool call -- shared by both the flat and the round-grouped rendering below
// so a tool's own row looks identical either way.
function TraceToolRow({ t }: { t: ToolUsage }) {
  return (
    <li>
      <span className="trace-tool-name">{t.name}</span>
      {t.checkpoint_id && <span className="trace-tool-checkpoint">{t.checkpoint_id}</span>}
      {Object.keys(t.params).length > 0 && <span className="trace-tool-params">{JSON.stringify(t.params)}</span>}
    </li>
  );
}

export function ExecutionTraceView({ trace }: { trace: ExecutionTrace }) {
  const [open, setOpen] = useState(true);

  // The backend's agentic loop (orchestrator/controller.py) can call more tools in a later round,
  // informed by what an earlier round found -- group by round only when that actually happened, so
  // the overwhelmingly common single-round query still reads as a plain flat list, not "Round 1"
  // labelling itself for no reason.
  const maxRound = trace.tools_used.reduce((m, t) => Math.max(m, t.round), 1);
  const rounds =
    maxRound > 1
      ? Array.from({ length: maxRound }, (_, i) => i + 1)
          .map((r) => ({ round: r, tools: trace.tools_used.filter((t) => t.round === r) }))
          .filter((g) => g.tools.length > 0)
      : null;

  return (
    <div className="trace">
      <button className="trace-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="trace-toggle-label">Execution trace</span>
        <span className="trace-toggle-icon">{open ? "–" : "+"}</span>
      </button>
      {open && (
        <div className="trace-body">
          <div className="trace-field">
            <dt>Task</dt>
            <dd>{trace.selected_task}</dd>
          </div>

          <div className="trace-field">
            <dt>Tools</dt>
            <dd>
              {trace.tools_used.length === 0 ? (
                <span className="trace-empty">none invoked</span>
              ) : rounds ? (
                <div className="trace-rounds">
                  {rounds.map((g) => (
                    <div className="trace-round" key={g.round}>
                      <span className="trace-round-label">
                        Round {g.round}
                        {g.round > 1 && <span className="trace-round-hint"> · informed by earlier results</span>}
                      </span>
                      <ul className="trace-tool-list">
                        {g.tools.map((t, i) => (
                          <TraceToolRow t={t} key={i} />
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              ) : (
                <ul className="trace-tool-list">
                  {trace.tools_used.map((t, i) => (
                    <TraceToolRow t={t} key={i} />
                  ))}
                </ul>
              )}
            </dd>
          </div>

          {trace.warnings.length > 0 && (
            <div className="trace-field trace-field-warning">
              <dt>Warnings</dt>
              <dd>
                <ul>
                  {trace.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </dd>
            </div>
          )}

          <div className="trace-field">
            <dt>Input</dt>
            <dd>
              <pre>{trace.input_summary}</pre>
            </dd>
          </div>

          <div className="trace-timestamp">{trace.timestamp}</div>
        </div>
      )}
    </div>
  );
}
