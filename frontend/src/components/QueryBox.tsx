import { useEffect, useRef, useState } from "react";
import { SendIcon } from "./icons";

// Short labels, full prompts -- from problem_statement.txt's "Representative Queries", plus one
// each for the groundwater and wildfire tools. Some target specialists that don't exist yet;
// those come back as "no tool matched", which is a legitimate (and visible) outcome.
const EXAMPLES: { label: string; prompt: string }[] = [
  { label: "Water body", prompt: "Highlight the water body referred to in the query." },
  { label: "Well siting", prompt: "Should I dig a well/tubewell at this location?" },
  { label: "Land cover", prompt: "Describe the land-cover and major objects visible in this image." },
  { label: "Change", prompt: "What changed between these two dates, and where did the change occur?" },
  {
    label: "Optical + SAR",
    prompt: "Use the optical and SAR images together to identify built-up and water-covered regions.",
  },
  { label: "Built-up trend", prompt: "Has the built-up area increased, decreased, or remained unchanged?" },
  { label: "Wildfire", prompt: "Is there any active wildfire near this location?" },
];

const MAX_TEXTAREA_HEIGHT_PX = 160;

interface Props {
  hasUpload: boolean;
  hasLocation: boolean;
  submitting: boolean;
  showExamples: boolean;
  onSubmit: (queryText: string) => void;
}

export function QueryBox({ hasUpload, hasLocation, submitting, showExamples, onSubmit }: Props) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow with content, capped so a long paste doesn't swallow the screen.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT_PX)}px`;
  }, [text]);

  const canSubmit = text.trim().length > 0 && (hasUpload || hasLocation) && !submitting;

  const submit = () => {
    if (!canSubmit) return;
    onSubmit(text.trim());
    setText("");
  };

  return (
    <div className="composer-inner">
      {showExamples && (
        <div className="chip-row">
          {EXAMPLES.map((e) => (
            <button key={e.label} className="chip" onClick={() => setText(e.prompt)} title={e.prompt}>
              {e.label}
            </button>
          ))}
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
          value={text}
          onChange={(e) => setText(e.target.value)}
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
