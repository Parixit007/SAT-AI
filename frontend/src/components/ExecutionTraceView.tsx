import { useState } from "react";
import type { ExecutionTrace } from "../api/client";

export function ExecutionTraceView({ trace }: { trace: ExecutionTrace }) {
  const [open, setOpen] = useState(true);

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
              ) : (
                <ul className="trace-tool-list">
                  {trace.tools_used.map((t, i) => (
                    <li key={i}>
                      <span className="trace-tool-name">{t.name}</span>
                      {t.checkpoint_id && <span className="trace-tool-checkpoint">{t.checkpoint_id}</span>}
                      {Object.keys(t.params).length > 0 && (
                        <span className="trace-tool-params">{JSON.stringify(t.params)}</span>
                      )}
                    </li>
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
