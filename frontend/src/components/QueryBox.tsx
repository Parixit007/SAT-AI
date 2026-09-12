import { useEffect, useState } from "react";
import type { LocationIn } from "../api/client";

// Short labels, full prompts -- the full sentences were stacking three lines deep in the panel and
// crowding out the controls underneath. From problem_statement.txt's "Representative Queries",
// plus one for the groundwater tool. Some target specialists that don't exist yet; those come back
// as "no tool matched", which is a legitimate (and visible) outcome.
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
];

interface Props {
  hasUpload: boolean;
  submitting: boolean;
  location: LocationIn | null;
  onLocationChange: (loc: LocationIn | null) => void;
  onSubmit: (queryText: string) => void;
}

export function QueryBox({ hasUpload, submitting, location, onLocationChange, onSubmit }: Props) {
  const [text, setText] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");

  // Keep the numeric fields in step with map clicks, without clobbering what someone is mid-way
  // through typing (the map is the source of truth only when it's the thing that changed).
  useEffect(() => {
    if (location) {
      setLat(location.lat.toFixed(5));
      setLon(location.lon.toFixed(5));
    }
  }, [location?.lat, location?.lon]);

  const commitTypedCoords = (nextLat: string, nextLon: string) => {
    const parsedLat = Number(nextLat);
    const parsedLon = Number(nextLon);
    if (nextLat.trim() === "" || nextLon.trim() === "" || Number.isNaN(parsedLat) || Number.isNaN(parsedLon)) {
      if (nextLat.trim() === "" && nextLon.trim() === "") onLocationChange(null);
      return;
    }
    onLocationChange({ lat: parsedLat, lon: parsedLon });
  };

  const canSubmit = text.trim().length > 0 && (hasUpload || location !== null) && !submitting;

  return (
    <div className="query-block">
      <textarea
        rows={3}
        placeholder="Ask about the imagery or the selected location…"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="chip-row">
        {EXAMPLES.map((e) => (
          <button key={e.label} className="chip" onClick={() => setText(e.prompt)} title={e.prompt}>
            {e.label}
          </button>
        ))}
      </div>

      <div className="coord-field">
        <div className="coord-inputs">
          <input
            aria-label="Latitude"
            type="number"
            step="any"
            placeholder="lat"
            value={lat}
            onChange={(e) => {
              setLat(e.target.value);
              commitTypedCoords(e.target.value, lon);
            }}
          />
          <input
            aria-label="Longitude"
            type="number"
            step="any"
            placeholder="lon"
            value={lon}
            onChange={(e) => {
              setLon(e.target.value);
              commitTypedCoords(lat, e.target.value);
            }}
          />
          {location && (
            <button
              className="coord-clear"
              onClick={() => {
                setLat("");
                setLon("");
                onLocationChange(null);
              }}
              title="Clear location"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      <button className="run-button" disabled={!canSubmit} onClick={() => onSubmit(text.trim())}>
        {submitting ? "Running" : "Run query"}
      </button>

      {!hasUpload && !location && (
        <p className="rail-hint">Click the map or add imagery to enable the query.</p>
      )}
    </div>
  );
}
