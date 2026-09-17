import { useEffect, useState } from "react";
import { captureArea, type LocationIn, type UploadResponse } from "../api/client";
import { errorMessage } from "../errorMessage";
import { XIcon } from "./icons";
import { MapPicker } from "./MapPicker";

interface Props {
  open: boolean;
  location: LocationIn | null;
  radiusM: number;
  onPick: (loc: LocationIn) => void;
  onLocationChange: (loc: LocationIn | null) => void;
  onClose: () => void;
  /** Called with a successful area capture, same shape as a real file upload -- the caller
   *  (App.tsx) already has a handler for exactly this (handleUploaded), reused as-is. */
  onCaptured: (result: UploadResponse) => void;
}

function approxAreaSizeKm(corners: LocationIn[]): { widthKm: number; heightKm: number } | null {
  if (corners.length !== 2) return null;
  const [a, b] = corners;
  const meanLatRad = ((a.lat + b.lat) / 2) * (Math.PI / 180);
  const KM_PER_LAT_DEG = 111.32;
  const widthKm = Math.abs(b.lon - a.lon) * KM_PER_LAT_DEG * Math.cos(meanLatRad);
  const heightKm = Math.abs(b.lat - a.lat) * KM_PER_LAT_DEG;
  return { widthKm, heightKm };
}

// Always mounted (visually slid off-screen when closed via CSS transform), not conditionally
// rendered -- Leaflet doesn't like being torn down and reinitialized on every open, and this way
// the tile cache stays warm between opens.
export function MapDrawer({ open, location, radiusM, onPick, onLocationChange, onClose, onCaptured }: Props) {
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [mode, setMode] = useState<"pick" | "select-area">("pick");
  const [corners, setCorners] = useState<LocationIn[]>([]);
  const [capturing, setCapturing] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);

  useEffect(() => {
    if (location) {
      setLat(location.lat.toFixed(5));
      setLon(location.lon.toFixed(5));
    } else {
      setLat("");
      setLon("");
    }
  }, [location?.lat, location?.lon]);

  const commitTypedCoords = (nextLat: string, nextLon: string) => {
    const parsedLat = Number(nextLat);
    const parsedLon = Number(nextLon);
    if (nextLat.trim() === "" || nextLon.trim() === "" || Number.isNaN(parsedLat) || Number.isNaN(parsedLon)) {
      onLocationChange(null);
      return;
    }
    onLocationChange({ lat: parsedLat, lon: parsedLon });
  };

  const changeMode = (next: "pick" | "select-area") => {
    setMode(next);
    setCorners([]);
    setCaptureError(null);
  };

  const handleCapture = async () => {
    if (corners.length !== 2) return;
    const [a, b] = corners;
    setCapturing(true);
    setCaptureError(null);
    try {
      const result = await captureArea({
        min_lat: Math.min(a.lat, b.lat),
        min_lon: Math.min(a.lon, b.lon),
        max_lat: Math.max(a.lat, b.lat),
        max_lon: Math.max(a.lon, b.lon),
      });
      onCaptured(result);
      setCorners([]);
      setMode("pick");
      onClose();
    } catch (err) {
      setCaptureError(errorMessage(err));
    } finally {
      setCapturing(false);
    }
  };

  const size = approxAreaSizeKm(corners);

  return (
    <>
      <div className={`drawer-backdrop ${open ? "is-open" : ""}`} onClick={onClose} aria-hidden={!open} />
      <aside className={`map-drawer ${open ? "is-open" : ""}`} aria-hidden={!open}>
        <div className="drawer-header">
          <h2>Location</h2>
          <button className="icon-btn drawer-close" onClick={onClose} aria-label="Close map">
            <XIcon />
          </button>
        </div>

        <div className="drawer-mode-toggle" role="tablist">
          <button
            role="tab"
            aria-selected={mode === "pick"}
            className={mode === "pick" ? "is-active" : ""}
            onClick={() => changeMode("pick")}
          >
            Pick location
          </button>
          <button
            role="tab"
            aria-selected={mode === "select-area"}
            className={mode === "select-area" ? "is-active" : ""}
            onClick={() => changeMode("select-area")}
          >
            Select area
          </button>
        </div>

        <div className="drawer-map-wrap">
          <MapPicker
            location={location}
            radiusM={radiusM}
            onPick={onPick}
            mode={mode}
            corners={corners}
            onCornersChange={setCorners}
          />
        </div>

        {mode === "pick" ? (
          <div className="drawer-coords">
            <label className="drawer-field">
              <span>Latitude</span>
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
            </label>
            <label className="drawer-field">
              <span>Longitude</span>
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
            </label>
            {location && (
              <button
                className="drawer-clear"
                onClick={() => {
                  setLat("");
                  setLon("");
                  onLocationChange(null);
                }}
              >
                Clear location
              </button>
            )}
          </div>
        ) : (
          <div className="drawer-coords">
            <p className="drawer-hint">
              Click two opposite corners on the map to select an area, then capture a real
              satellite image of it — ready to query immediately, no file needed.
            </p>
            {size && (
              <p className="drawer-area-size">
                ~{size.widthKm.toFixed(size.widthKm < 1 ? 3 : 1)} km × {size.heightKm.toFixed(size.heightKm < 1 ? 3 : 1)} km
              </p>
            )}
            {captureError && <p className="error-text">{captureError}</p>}
            <div className="drawer-capture-actions">
              <button className="drawer-clear" onClick={() => setCorners([])} disabled={corners.length === 0 || capturing}>
                Clear selection
              </button>
              <button className="capture-btn" onClick={handleCapture} disabled={corners.length !== 2 || capturing}>
                {capturing ? "Capturing…" : "Capture image"}
              </button>
            </div>
          </div>
        )}
      </aside>
    </>
  );
}
