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
 *  image supplying its own georeferencing. Without this the pin can land off-screen.
 *
 *  The jump is often large (default view is zoom 4, most picks recenter to 11 -- a 7-level zoom
 *  across a big geographic distance) while three tile layers are stacked on top of each other.
 *  At the old 0.6s duration this was empirically confirmed jittery: a single click fired over a
 *  thousand tile requests across the animated path in well under a second, well more than the
 *  browser could decode/paint smoothly. `duration`/`easeLinearity` alone don't fix that -- the
 *  actual fix is each TileLayer's own `updateWhenZooming={false}` below, which stops Leaflet from
 *  swapping in new tiles at every intermediate zoom level *during* the animation (the documented
 *  cause of exactly this symptom) and only updates once it settles. The longer duration and eased
 *  curve here just make the settled motion itself read as a deliberate pan, not a snap. */
function Recenter({ location }: { location: LocationIn | null }) {
  const map = useMap();
  useEffect(() => {
    if (location) {
      map.flyTo([location.lat, location.lon], Math.max(map.getZoom(), 11), {
        duration: 1.3,
        easeLinearity: 0.15,
      });
    }
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
      >
        {/* Esri World Imagery -- same free, keyless satellite basemap as before (already
            sub-meter resolution in most areas; there isn't a meaningfully higher-resolution
            free alternative that doesn't require an API key). The two reference layers below
            are what's new: Esri's own free overlays, designed specifically to sit on top of
            World Imagery for exactly this hybrid look -- roads underneath, place labels and
            political borders on top so text stays legible over the road lines. */}
        {/* updateWhenZooming={false} on all three stacked layers is the actual fix for the
            reported map jitter -- without it, each layer independently swaps in new tiles at
            every intermediate zoom level while flyTo is still animating, which is expensive
            times three and reads as stutter. With it, each layer waits for the animation to
            settle before updating, so the animation itself stays smooth and tiles pop in once,
            after motion stops, instead of mid-flight. */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          maxZoom={18}
          zIndex={1}
          updateWhenZooming={false}
        />
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}"
          maxZoom={18}
          zIndex={2}
          updateWhenZooming={false}
        />
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          maxZoom={18}
          zIndex={3}
          updateWhenZooming={false}
          attribution="Tiles &copy; Esri &mdash; Esri, Maxar, Earthstar Geographics, and the GIS User Community"
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
