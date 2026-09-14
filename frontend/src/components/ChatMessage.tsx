import type { QueryResponse } from "../api/client";
import { ResultsView } from "./ResultsView";

export type ChatMsg =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; status: "pending" }
  | { id: string; role: "assistant"; status: "done"; result: QueryResponse }
  | { id: string; role: "assistant"; status: "error"; error: string };

export function ChatMessage({ message }: { message: ChatMsg }) {
  if (message.role === "user") {
    return (
      <div className="msg-row user">
        <div className="msg-bubble user">{message.text}</div>
      </div>
    );
  }

  if (message.status === "pending") {
    return (
      <div className="msg-row assistant">
        <div className="msg-bubble assistant pending">
          <span className="typing-dots" aria-label="Thinking">
            <span></span>
            <span></span>
            <span></span>
          </span>
        </div>
      </div>
    );
  }

  if (message.status === "error") {
    return (
      <div className="msg-row assistant">
        <div className="msg-bubble assistant error">{message.error}</div>
      </div>
    );
  }

  return (
    <div className="msg-row assistant">
      <div className="msg-bubble assistant result-bubble">
        <ResultsView result={message.result} />
      </div>
    </div>
  );
}
