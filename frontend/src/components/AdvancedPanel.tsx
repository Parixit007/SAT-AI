import type { ForcedToolCall, ToolSpecOut } from "../api/client";
import { TOOL_LABEL } from "../toolMeta";

// Manual override for the automatic LLM tool-routing: pick exactly which specialist(s) run and
// tune their arguments directly, instead of trusting auto-selection every time. `value === null`
// (the default) means fully automatic, unchanged; a non-null (possibly empty) array means "run
// exactly these tools" -- see backend/app/orchestrator/controller.py's forced_tool_calls.
interface Props {
  tools: ToolSpecOut[];
  value: ForcedToolCall[] | null;
  onChange: (value: ForcedToolCall[] | null) => void;
}

function ParamField({
  name,
  schema,
  value,
  onChange,
}: {
  name: string;
  schema: { type: string; description?: string };
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (schema.type === "boolean") {
    return (
      <label className="param-field param-field-checkbox">
        <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        {name}
      </label>
    );
  }
  if (schema.type === "integer" || schema.type === "number") {
    return (
      <label className="param-field">
        <span>{name}</span>
        <input
          type="number"
          value={value === undefined || value === null ? "" : String(value)}
          placeholder={schema.description}
          onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
        />
      </label>
    );
  }
  if (schema.type === "string") {
    return (
      <label className="param-field">
        <span>{name}</span>
        <input
          type="text"
          value={typeof value === "string" ? value : ""}
          placeholder={schema.description}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
    );
  }
  // Anything the generic cases above don't cover (nested object/array params) -- none of the 7
  // current tool schemas need this, but a raw-JSON fallback means a future one is never unrepresentable.
  return (
    <label className="param-field">
      <span>{name} (JSON)</span>
      <textarea
        rows={2}
        defaultValue={JSON.stringify(value ?? {})}
        onBlur={(e) => {
          try {
            onChange(JSON.parse(e.target.value));
          } catch {
            // Invalid JSON mid-edit -- leave the last valid value in place rather than clearing it.
          }
        }}
      />
    </label>
  );
}

export function AdvancedPanel({ tools, value, onChange }: Props) {
  const active = value !== null;
  const selectedNames = new Set((value ?? []).map((c) => c.tool_name));

  const toggleTool = (tool: ToolSpecOut) => {
    const current = value ?? [];
    if (selectedNames.has(tool.name)) {
      onChange(current.filter((c) => c.tool_name !== tool.name));
    } else {
      onChange([...current, { tool_name: tool.name, arguments: {} }]);
    }
  };

  const setArg = (toolName: string, argName: string, argValue: unknown) => {
    onChange(
      (value ?? []).map((c) =>
        c.tool_name === toolName ? { ...c, arguments: { ...c.arguments, [argName]: argValue } } : c
      )
    );
  };

  return (
    <div className="advanced-panel">
      <label className="advanced-toggle">
        <input type="checkbox" checked={active} onChange={(e) => onChange(e.target.checked ? [] : null)} />
        Manually choose tools instead of automatic routing
      </label>

      {active && (
        <div className="advanced-tool-list">
          {tools.map((tool) => {
            const selected = selectedNames.has(tool.name);
            const forcedCall = (value ?? []).find((c) => c.tool_name === tool.name);
            const params = Object.entries(tool.parameters_schema.properties ?? {});
            return (
              <div key={tool.name} className={`advanced-tool ${selected ? "is-selected" : ""}`}>
                <button className="chip advanced-tool-chip" onClick={() => toggleTool(tool)} aria-pressed={selected}>
                  {TOOL_LABEL[tool.name] ?? tool.name}
                </button>
                {selected && params.length > 0 && (
                  <div className="advanced-tool-params">
                    {params.map(([name, schema]) => (
                      <ParamField
                        key={name}
                        name={name}
                        schema={schema}
                        value={forcedCall?.arguments[name]}
                        onChange={(v) => setArg(tool.name, name, v)}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          {selectedNames.size === 0 && (
            <p className="advanced-hint">No tools picked yet -- the query will run nothing until you pick at least one.</p>
          )}
        </div>
      )}
    </div>
  );
}
