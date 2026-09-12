import { useEffect } from "react";
import { Circle, MapContainer, Marker, TileLayer, useMap, useMapEvents } from "react-leaflet";
import { divIcon } from "leaflet";
import "leaflet/dist/leaflet.css";
import type { LocationIn } from "../api/client";

// Leaflet's default marker is a bundled PNG that breaks under Vite's asset handling. A divIcon
// sidesteps that entirely and lets the pin match the rest of the interface (see .map-pin in App.css).
const pinIcon = divIcon({
  className: "map-pin-wrapper",
  html: '<span class="map-pin"></span>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

function ClickHandler({ onPick }: { onPick: (loc: LocationIn) => void }) {
  useMapEvents({
    click(e) {
      onPick({ lat: e.latlng.lat, lon: e.latlng.lng });
    },
  });
  return null;
}

/** Recenters when the location is set from outside the map -- typing coordinates, or an uploaded
 *  image supplying its own georeferencing. Without this the pin can land off-screen. */
function Recenter({ location }: { location: LocationIn | null }) {
  const map = useMap();
  useEffect(() => {
    if (location) map.flyTo([location.lat, location.lon], Math.max(map.getZoom(), 11), { duration: 0.6 });
  }, [location?.lat, location?.lon]);
  return null;
}

interface Props {
  location: LocationIn | null;
  radiusM: number;
  onPick: (loc: LocationIn) => void;
}

export function MapPicker({ location, radiusM, onPick }: Props) {
  return (
    <div className="map-shell">
      <MapContainer
        center={[20.5937, 78.9629]}
        zoom={4}
        className="map-canvas"
        worldCopyJump
        attributionControl={false}
      >
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          maxZoom={18}
        />
        <ClickHandler onPick={onPick} />
        <Recenter location={location} />
        {location && (
          <>
            <Circle
              center={[location.lat, location.lon]}
              radius={radiusM}
              pathOptions={{ color: "#4fa8d8", weight: 1, fillColor: "#4fa8d8", fillOpacity: 0.12 }}
            />
            <Marker position={[location.lat, location.lon]} icon={pinIcon} />
          </>
        )}
      </MapContainer>

      <div className="map-readout">
        {location ? (
          <>
            <span className="map-readout-coords">
              {location.lat.toFixed(5)}, {location.lon.toFixed(5)}
            </span>
            <span className="map-readout-radius">{(radiusM / 1000).toFixed(1)} km radius</span>
          </>
        ) : (
          <span className="map-readout-empty">Click the map to set a location</span>
        )}
      </div>
    </div>
  );
}
