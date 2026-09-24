import { useEffect, useRef, useState } from "react";
import "./App.css";
import {
  listTools,
  runQuery,
  type ChatDetail,
  type ForcedToolCall,
  type LocationIn,
  type ToolSpecOut,
  type UploadResponse,
} from "./api/client";
import { AdvancedPanel } from "./components/AdvancedPanel";
import { CapabilitiesGallery } from "./components/CapabilitiesGallery";
import { ChatHistoryDrawer } from "./components/ChatHistoryDrawer";
import { HistoryIcon, MapPinIcon, PaperclipIcon, PlusIcon, SlidersIcon, XIcon } from "./components/icons";
import { MapDrawer } from "./components/MapDrawer";
import { QueryBox } from "./components/QueryBox";
import { QueryEntry, type QueryEntryState } from "./components/QueryEntry";
import { UploadPanel } from "./components/UploadPanel";
import { EXAMPLE_PROMPT } from "./toolMeta";
import { errorMessage } from "./errorMessage";

const QUERY_RADIUS_M = 1000;

function App() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [location, setLocation] = useState<LocationIn | null>(null);
  const [entries, setEntries] = useState<QueryEntryState[]>([]);
  const [chatId, setChatId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [mapOpen, setMapOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [attachOpen, setAttachOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [queryText, setQueryText] = useState("");
  const [tools, setTools] = useState<ToolSpecOut[]>([]);
  const [forcedTools, setForcedTools] = useState<ForcedToolCall[] | null>(null);
  const scrollBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollBottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length]);

  useEffect(() => {
    // Drives the homepage gallery and the Advanced panel from one live source -- if this fails
    // (backend not up yet), both simply render with no tools rather than crashing the app; the
    // rest of the UI (upload, map, manual query text) still works.
    listTools()
      .then(setTools)
      .catch(() => setTools([]));
  }, []);

  const hasUpload = !!upload && upload.images.length > 0;

  const runOneQuery = async (text: string) => {
    const id = crypto.randomUUID();
    setEntries((e) => [...e, { id, queryText: text, status: "pending" }]);
    setSubmitting(true);
    try {
      const result = await runQuery(upload?.input_id ?? null, text, location, forcedTools, chatId);
      setChatId(result.chat_id);
      setEntries((e) => e.map((entry) => (entry.id === id ? { id, queryText: text, status: "done", result } : entry)));
    } catch (err) {
      const error = errorMessage(err);
      setEntries((e) => e.map((entry) => (entry.id === id ? { id, queryText: text, status: "error", error } : entry)));
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmit = (text: string) => {
    void runOneQuery(text);
  };

  const handleRetry = (text: string) => {
    void runOneQuery(text);
  };

  const handleNewChat = () => {
    setEntries([]);
    setChatId(null);
  };

  const handleSelectChat = (detail: ChatDetail) => {
    setEntries(
      detail.entries.map((e) => ({
        id: crypto.randomUUID(),
        queryText: e.query_text,
        status: "done" as const,
        result: e.response,
      })),
    );
    setChatId(detail.id);
  };

  const handleUploaded = (res: UploadResponse) => {
    setUpload(res);
    setAttachOpen(false);
    // A georeferenced upload already knows where it is -- drop the pin there so the map and any
    // location-based tool agree without the user retyping coordinates.
    const geo = res.images.find((i) => i.geo)?.geo;
    if (geo) setLocation({ lat: geo.center_lat, lon: geo.center_lon });
  };

  const handlePickCapability = (tool: ToolSpecOut) => {
    setQueryText(EXAMPLE_PROMPT[tool.name] ?? tool.description);
    if (tool.uses_images && !hasUpload) setAttachOpen(true);
    else if (tool.requires_location && !location) setMapOpen(true);
  };

  return (
    <div className="app-shell">
      <div className="bg-blobs" aria-hidden="true">
        <div className="blob blob-water" />
        <div className="blob blob-land" />
        <div className="blob blob-violet" />
      </div>

      <header className="chat-header">
        <div className="brand">
          <h1>SatQuery AI</h1>
          <span className="brand-tag">Agentic analysis for remote-sensing imagery</span>
        </div>
        <div className="header-actions">
          <button
            className="btn-icon"
            onClick={() => setHistoryOpen(true)}
            aria-label="Chat history"
            aria-pressed={historyOpen}
            title="Chat history"
          >
            <HistoryIcon size={15} />
          </button>
          {entries.length > 0 && (
            <button className="btn btn-text" onClick={handleNewChat} title="Start a new chat">
              <PlusIcon size={13} /> New chat
            </button>
          )}
        </div>
      </header>

      <main className="chat-main">
        <div className="chat-scroll">
          {entries.length === 0 ? (
            <CapabilitiesGallery tools={tools} onPick={handlePickCapability} />
          ) : (
            entries.map((entry) => <QueryEntry key={entry.id} entry={entry} onRetry={handleRetry} />)
          )}
          <div ref={scrollBottomRef} />
        </div>
      </main>

      <footer className="composer">
        {hasUpload && (
          <div className="attach-chip-row">
            {upload!.images.map((img) => (
              // stored_path, not filename -- two uploaded files can share a filename (e.g. an
              // optical+SAR pair both called "export.tif"), which isn't a unique key on its own.
              <span key={img.stored_path} className="attach-chip">
                {img.filename}
                <span className="attach-chip-meta">{img.modality_guess}</span>
              </span>
            ))}
          </div>
        )}
        {upload && (upload.errors.length > 0 || upload.warnings.length > 0) && (
          <ul className="warning-list">
            {[...upload.errors, ...upload.warnings].map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        )}
        {location && (
          <div className="location-chip-row">
            <span className="location-chip">
              <MapPinIcon size={13} />
              {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
              <button onClick={() => setLocation(null)} aria-label="Clear location">
                <XIcon size={12} />
              </button>
            </span>
          </div>
        )}
        {forcedTools !== null && (
          <div className="location-chip-row">
            <span className="location-chip advanced-active-chip">
              <SlidersIcon size={13} />
              Manual: {forcedTools.length === 0 ? "no tools picked" : forcedTools.map((t) => t.tool_name.replace(/_/g, " ")).join(", ")}
              <button onClick={() => setForcedTools(null)} aria-label="Return to automatic tool selection">
                <XIcon size={12} />
              </button>
            </span>
          </div>
        )}

        <div className="composer-bar">
          <div className="composer-attach">
            <button
              className={`icon-btn attach-btn ${attachOpen ? "is-active" : ""}`}
              onClick={() => setAttachOpen((v) => !v)}
              aria-label="Attach imagery"
              aria-pressed={attachOpen}
            >
              <PaperclipIcon />
            </button>
            {attachOpen && (
              <>
                <div className="popover-backdrop" onClick={() => setAttachOpen(false)} />
                <div className="attach-popover">
                  <UploadPanel onUploaded={handleUploaded} />
                </div>
              </>
            )}
          </div>

          <div className="composer-attach">
            <button
              className={`icon-btn attach-btn ${advancedOpen || forcedTools !== null ? "is-active" : ""}`}
              onClick={() => setAdvancedOpen((v) => !v)}
              aria-label="Advanced: manual tool selection"
              aria-pressed={advancedOpen}
              title="Advanced: manual tool selection"
            >
              <SlidersIcon />
            </button>
            {advancedOpen && (
              <>
                <div className="popover-backdrop" onClick={() => setAdvancedOpen(false)} />
                <div className="attach-popover advanced-popover">
                  <AdvancedPanel tools={tools} value={forcedTools} onChange={setForcedTools} />
                </div>
              </>
            )}
          </div>

          <QueryBox
            value={queryText}
            onChange={setQueryText}
            tools={tools}
            hasUpload={hasUpload}
            hasLocation={location !== null}
            submitting={submitting}
            showExamples={entries.length === 0}
            onSubmit={handleSubmit}
          />
        </div>
      </footer>

      <button
        className={`icon-btn map-fab ${mapOpen ? "is-active" : ""}`}
        onClick={() => setMapOpen((v) => !v)}
        aria-label={mapOpen ? "Close map" : "Open map"}
        aria-pressed={mapOpen}
      >
        <MapPinIcon size={22} />
        {location && <span className="fab-dot" />}
      </button>

      <MapDrawer
        open={mapOpen}
        location={location}
        radiusM={QUERY_RADIUS_M}
        onPick={setLocation}
        onLocationChange={setLocation}
        onClose={() => setMapOpen(false)}
        onCaptured={handleUploaded}
      />

      <ChatHistoryDrawer
        open={historyOpen}
        activeChatId={chatId}
        onClose={() => setHistoryOpen(false)}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onActiveChatDeleted={handleNewChat}
      />
    </div>
  );
}

export default App;
