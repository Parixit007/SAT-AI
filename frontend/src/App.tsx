import { useEffect, useRef, useState } from "react";
import "./App.css";
import { runQuery, type LocationIn, type UploadResponse } from "./api/client";
import { ChatMessage, type ChatMsg } from "./components/ChatMessage";
import { MapPinIcon, PaperclipIcon, XIcon } from "./components/icons";
import { MapDrawer } from "./components/MapDrawer";
import { QueryBox } from "./components/QueryBox";
import { UploadPanel } from "./components/UploadPanel";
import { errorMessage } from "./errorMessage";

const QUERY_RADIUS_M = 1000;

function App() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [location, setLocation] = useState<LocationIn | null>(null);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [mapOpen, setMapOpen] = useState(false);
  const [attachOpen, setAttachOpen] = useState(false);
  const scrollBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollBottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  const handleQuery = async (queryText: string) => {
    const userMsgId = crypto.randomUUID();
    const pendingId = crypto.randomUUID();
    setMessages((m) => [
      ...m,
      { id: userMsgId, role: "user", text: queryText },
      { id: pendingId, role: "assistant", status: "pending" },
    ]);
    setSubmitting(true);
    try {
      const result = await runQuery(upload?.input_id ?? null, queryText, location);
      setMessages((m) =>
        m.map((msg) => (msg.id === pendingId ? { id: pendingId, role: "assistant", status: "done", result } : msg)),
      );
    } catch (err) {
      const error = errorMessage(err);
      setMessages((m) =>
        m.map((msg) => (msg.id === pendingId ? { id: pendingId, role: "assistant", status: "error", error } : msg)),
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleUploaded = (res: UploadResponse) => {
    setUpload(res);
    setAttachOpen(false);
    // A georeferenced upload already knows where it is -- drop the pin there so the map and any
    // location-based tool agree without the user retyping coordinates.
    const geo = res.images.find((i) => i.geo)?.geo;
    if (geo) setLocation({ lat: geo.center_lat, lon: geo.center_lon });
  };

  const hasUpload = !!upload && upload.images.length > 0;

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
      </header>

      <main className="chat-main">
        <div className="chat-scroll">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🛰️</div>
              <h2>Ask about any location or image</h2>
              <p>Attach optical/SAR imagery with the clip icon, or open the map to drop a pin — then ask a question.</p>
            </div>
          ) : (
            messages.map((m) => <ChatMessage key={m.id} message={m} />)
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

          <QueryBox
            hasUpload={hasUpload}
            hasLocation={location !== null}
            submitting={submitting}
            showExamples={messages.length === 0}
            onSubmit={handleQuery}
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
    </div>
  );
}

export default App;
