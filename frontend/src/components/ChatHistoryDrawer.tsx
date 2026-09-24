import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { deleteChat, getChat, listChats, type ChatDetail, type ChatSummary } from "../api/client";
import { errorMessage } from "../errorMessage";
import { PlusIcon, TrashIcon, XIcon } from "./icons";

interface Props {
  open: boolean;
  activeChatId: string | null;
  onClose: () => void;
  /** Called with the fully-loaded chat once fetched -- App.tsx only has to convert it into entries. */
  onSelectChat: (detail: ChatDetail) => void;
  onNewChat: () => void;
  /** The chat currently open in the main view was just deleted from in here -- caller should reset
   *  to a blank/new-chat state rather than keep sending queries against a chat_id that's now gone. */
  onActiveChatDeleted: () => void;
}

function formatRelativeTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const diffMin = Math.round((Date.now() - date.getTime()) / 60000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.round(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// Always mounted, slid off-screen via CSS transform when closed -- same pattern as MapDrawer, so
// the open/close transition is a transform change, not a mount/unmount.
export function ChatHistoryDrawer({ open, activeChatId, onClose, onSelectChat, onNewChat, onActiveChatDeleted }: Props) {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [openingId, setOpeningId] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setConfirmId(null);
      return;
    }
    setLoading(true);
    setError(null);
    listChats()
      .then(setChats)
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setLoading(false));
  }, [open]);

  const handleOpen = async (id: string) => {
    setOpeningId(id);
    setError(null);
    try {
      const detail = await getChat(id);
      onSelectChat(detail);
      onClose();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setOpeningId(null);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteChat(id);
      setChats((cs) => cs.filter((c) => c.id !== id));
      setConfirmId(null);
      if (id === activeChatId) onActiveChatDeleted();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  return (
    <>
      <div className={`drawer-backdrop ${open ? "is-open" : ""}`} onClick={onClose} aria-hidden={!open} />
      <aside className={`history-drawer ${open ? "is-open" : ""}`} aria-hidden={!open}>
        <div className="drawer-header">
          <h2>Chat history</h2>
          <button className="icon-btn drawer-close" onClick={onClose} aria-label="Close chat history">
            <XIcon />
          </button>
        </div>

        <div className="history-new-chat-row">
          <button
            className="btn btn-text history-new-chat"
            onClick={() => {
              onNewChat();
              onClose();
            }}
          >
            <PlusIcon size={13} /> New chat
          </button>
        </div>

        <div className="history-list">
          {loading && <p className="history-empty">Loading…</p>}
          {!loading && error && <p className="error-text history-empty">{error}</p>}
          {!loading && !error && chats.length === 0 && <p className="history-empty">No past chats yet -- ask something to start one.</p>}
          {!loading &&
            chats.map((chat) => (
              <div key={chat.id} className={`history-row ${chat.id === activeChatId ? "is-active" : ""}`}>
                <button className="history-row-main" onClick={() => handleOpen(chat.id)} disabled={openingId === chat.id}>
                  <span className="history-row-title">{openingId === chat.id ? "Opening…" : chat.title}</span>
                  <span className="history-row-meta">
                    {formatRelativeTime(chat.updated_at)} · {chat.entry_count} {chat.entry_count === 1 ? "query" : "queries"}
                  </span>
                </button>

                <AnimatePresence initial={false} mode="wait">
                  {confirmId === chat.id ? (
                    <motion.div
                      key="confirm"
                      className="history-row-confirm"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.14 }}
                    >
                      <button className="btn btn-text history-confirm-yes" onClick={() => handleDelete(chat.id)}>
                        Delete
                      </button>
                      <button className="btn btn-text" onClick={() => setConfirmId(null)}>
                        Cancel
                      </button>
                    </motion.div>
                  ) : (
                    <motion.button
                      key="trash"
                      className="btn-icon history-row-delete"
                      onClick={() => setConfirmId(chat.id)}
                      aria-label={`Delete chat "${chat.title}"`}
                      title="Delete chat"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.14 }}
                    >
                      <TrashIcon size={14} />
                    </motion.button>
                  )}
                </AnimatePresence>
              </div>
            ))}
        </div>
      </aside>
    </>
  );
}
