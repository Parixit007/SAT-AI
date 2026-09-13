import { useState } from "react";
import "./App.css";
import { runQuery, type LocationIn, type QueryResponse, type UploadResponse } from "./api/client";
import { MapPicker } from "./components/MapPicker";
import { QueryBox } from "./components/QueryBox";
import { ResultsView } from "./components/ResultsView";
import { UploadPanel } from "./components/UploadPanel";

const QUERY_RADIUS_M = 1000;

function App() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [location, setLocation] = useState<LocationIn | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async (queryText: string) => {
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      setResult(await runQuery(upload?.input_id ?? null, queryText, location));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleUploaded = (res: UploadResponse) => {
    setUpload(res);
    setResult(null);
    setError(null);
    // A georeferenced upload already knows where it is -- drop the pin there so the map and any
    // location-based tool agree without the user retyping coordinates.
    const geo = res.images.find((i) => i.geo)?.geo;
    if (geo) setLocation({ lat: geo.center_lat, lon: geo.center_lon });
  };

  return (
    <div className="console">
      <header className="console-header">
        <h1>SatQuery AI</h1>
        <span className="brand-tag">Agentic analysis for remote-sensing imagery</span>
      </header>

      <div className="console-body">
        <aside className="panel">
          <section className="panel-section">
            <h2>Query</h2>
            <QueryBox
              hasUpload={!!upload && upload.images.length > 0}
              submitting={submitting}
              location={location}
              onLocationChange={setLocation}
              onSubmit={handleQuery}
            />
          </section>

          <section className="panel-section">
            <h2>Imagery</h2>
            <UploadPanel onUploaded={handleUploaded} />
            {upload && upload.images.length > 0 && (
              <ul className="upload-image-list">
                {upload.images.map((img) => (
                  // stored_path, not filename -- two uploaded files can share a filename (e.g. an
                  // optical+SAR pair both called "export.tif"), which isn't a unique key on its own.
                  <li key={img.stored_path}>
                    <span className="upload-filename">{img.filename}</span>
                    <span className="upload-meta">
                      {img.width}×{img.height} · {img.modality_guess}
                      {img.geo && ` · ${img.geo.center_lat.toFixed(3)}, ${img.geo.center_lon.toFixed(3)}`}
                    </span>
                  </li>
                ))}
              </ul>
            )}
            {upload && upload.errors.length > 0 && (
              <ul className="warning-list">
                {upload.errors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            )}
            {upload && upload.warnings.length > 0 && (
              <ul className="warning-list">
                {upload.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
          </section>

          {(error || result) && (
            <section className="panel-section panel-section-result">
              <h2>Result</h2>
              {error && <p className="error-text">{error}</p>}
              {result && <ResultsView result={result} />}
            </section>
          )}
        </aside>

        <main className="stage">
          <MapPicker location={location} radiusM={QUERY_RADIUS_M} onPick={setLocation} />
        </main>
      </div>
    </div>
  );
}

export default App;
