import { useEffect, useState } from "react";
import type { LocationIn } from "../api/client";
import { XIcon } from "./icons";
import { MapPicker } from "./MapPicker";

interface Props {
  open: boolean;
  location: LocationIn | null;
  radiusM: number;
  onPick: (loc: LocationIn) => void;
  onLocationChange: (loc: LocationIn | null) => void;
  onClose: () => void;
}

// Always mounted (visually slid off-screen when closed via CSS transform), not conditionally
// rendered -- Leaflet doesn't like being torn down and reinitialized on every open, and this way
// the tile cache stays warm between opens instead of refetching every time.
export function MapDrawer({ open, location, radiusM, onPick, onLocationChange, onClose }: Props) {
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");

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

        <div className="drawer-map-wrap">
          <MapPicker location={location} radiusM={radiusM} onPick={onPick} />
        </div>

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
      </aside>
    </>
  );
}
