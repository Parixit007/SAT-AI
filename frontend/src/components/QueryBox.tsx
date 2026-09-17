import { useEffect, useRef } from "react";
import type { ToolSpecOut } from "../api/client";
import { EXAMPLE_PROMPT } from "../toolMeta";
import { SendIcon } from "./icons";

const MAX_TEXTAREA_HEIGHT_PX = 160;

interface Props {
  value: string;
  onChange: (value: string) => void;
  tools: ToolSpecOut[];
  hasUpload: boolean;
  hasLocation: boolean;
  submitting: boolean;
  showExamples: boolean;
  onSubmit: (queryText: string) => void;
}

export function QueryBox({ value, onChange, tools, hasUpload, hasLocation, submitting, showExamples, onSubmit }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow with content, capped so a long paste doesn't swallow the screen.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT_PX)}px`;
  }, [value]);

  const canSubmit = value.trim().length > 0 && (hasUpload || hasLocation) && !submitting;

  const submit = () => {
    if (!canSubmit) return;
    onSubmit(value.trim());
    onChange("");
  };

  return (
    <div className="composer-inner">
      {showExamples && tools.length > 0 && (
        <div className="chip-row">
          {tools.map((t) => {
            const prompt = EXAMPLE_PROMPT[t.name] ?? t.description;
            return (
              <button key={t.name} className="chip" onClick={() => onChange(prompt)} title={prompt}>
                {t.name.replace(/_/g, " ")}
              </button>
            );
          })}
        </div>
      )}

      <div className="composer-row">
        <textarea
          ref={textareaRef}
          rows={1}
          placeholder={
            hasUpload || hasLocation
              ? "Ask about the imagery or the selected location…"
              : "Attach imagery or open the map to set a location…"
          }
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button className="send-btn" disabled={!canSubmit} onClick={submit} aria-label="Send query">
          <SendIcon />
        </button>
      </div>
    </div>
  );
}
