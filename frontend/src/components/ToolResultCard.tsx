import { AnimatePresence, motion } from "framer-motion";
import { Fragment, useState, type ReactNode } from "react";
import { evidenceImageUrl, type ToolResultOut } from "../api/client";
import { TOOL_LABEL } from "../toolMeta";
import { ChevronIcon } from "./icons";
import { DetectionTally, GroundingOverlay, type Detection } from "./GroundingOverlay";
import { LightboxImage } from "./Lightbox";

// Every card follows the same headline + optional expandable-detail shape: evidence imagery
// always shows (this app's core evidence-grounded-answer promise), only supplementary numbers/
// text collapse behind a toggle, collapsed by default -- keeps a multi-tool answer scannable
// instead of a wall of every field every specialist happened to compute.
function DetailToggle({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="tool-detail">
      <button className="btn btn-text tool-detail-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <ChevronIcon size={13} className={`chevron ${open ? "chevron-open" : ""}`} />
        {open ? "Hide details" : "Show details"}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
            style={{ overflow: "hidden" }}
          >
            <div className="tool-detail-body">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Bar({ label, value, percent = false }: { label: string; value: number | null | undefined; percent?: boolean }) {
  // A layer can be genuinely unavailable for a given location/image (e.g. groundwater's
  // soil_moisture coming back null when GEE has no coverage there) -- render that honestly
  // instead of crashing on null.toFixed(), per compute_groundwater_score()'s own "excluded, never
  // silently scored as 0" rule (see backend/app/gee/groundwater.py).
  if (value === null || value === undefined) {
    return (
      <div className="stat-bar">
        <div className="stat-bar-head">
          <span>{label}</span>
          <span className="data-value">unavailable</span>
        </div>
        <div className="stat-bar-track" />
      </div>
    );
  }
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="stat-bar">
      <div className="stat-bar-head">
        <span>{label}</span>
        <span className="data-value">{percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)}</span>
      </div>
      <div className="stat-bar-track">
        <div className="stat-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function EvidenceImage({ url, wide = false }: { url: string; wide?: boolean }) {
  return (
    <LightboxImage
      src={evidenceImageUrl(url)}
      alt="Visual evidence"
      className={`evidence-image tool-card-evidence ${wide ? "wide" : ""}`}
    />
  );
}

const GW_CATEGORY_CLASS: Record<string, string> = {
  "Very Low": "gw-very-low",
  Low: "gw-low",
  Moderate: "gw-moderate",
  High: "gw-high",
  "Very High": "gw-very-high",
};

function GroundwaterCard({ data, evidenceUrl }: { data: Record<string, unknown>; evidenceUrl: string | null }) {
  const category = String(data.category ?? "Unknown");
  const layers = (data.layers ?? {}) as Record<string, { raw: number | null; normalized: number | null }>;
  return (
    <div className="tool-card-body">
      <span className={`gw-badge ${GW_CATEGORY_CLASS[category] ?? ""}`}>{category}</span>
      {evidenceUrl && <EvidenceImage url={evidenceUrl} />}
      <DetailToggle>
        {Object.entries(layers).map(([name, v]) => (
          <Bar key={name} label={name.replace(/_/g, " ")} value={v.normalized} />
        ))}
      </DetailToggle>
    </div>
  );
}

const FIRE_STATUS_CLASS: Record<string, string> = {
  "No Fire Detected": "fire-none",
  "Possible Fire Activity": "fire-possible",
  "Active Fire Detected": "fire-active",
};

function WildfireCard({ data }: { data: Record<string, unknown> }) {
  const status = String(data.status ?? "Unknown");
  return (
    <div className="tool-card-body">
      <span className={`fire-badge ${FIRE_STATUS_CLASS[status] ?? ""}`}>{status}</span>
      <DetailToggle>
        <dl className="kv-grid">
          <dt>Detections</dt>
          <dd className="data-value">
            {String(data.detection_days ?? 0)} / {String(data.lookback_days ?? 0)} days
          </dd>
          <dt>Peak confidence</dt>
          <dd className="data-value">{String(data.max_confidence ?? "—")}</dd>
          <dt>Peak brightness</dt>
          <dd className="data-value">{data.max_brightness_k ? `${data.max_brightness_k} K` : "—"}</dd>
          <dt>Most recent</dt>
          <dd className="data-value">{String(data.most_recent_date ?? "—")}</dd>
        </dl>
      </DetailToggle>
    </div>
  );
}

function DetectionList({ detections }: { detections: Detection[] }) {
  return (
    <DetailToggle>
      <ul className="detection-list">
        {detections.map((d, i) => (
          <li key={i}>
            <span>{d.phrase}</span>
            <span className="data-value">
              {d.score.toFixed(2)} · [{d.bbox_xyxy.map((n) => Math.round(n)).join(", ")}]
            </span>
          </li>
        ))}
      </ul>
    </DetailToggle>
  );
}

function GroundingCard({ data, sourceImageUrl }: { data: Record<string, unknown>; sourceImageUrl: string | null }) {
  const detections = (data.detections ?? []) as Detection[];
  return (
    <div className="tool-card-body">
      <p className="detection-headline">
        <span className="detection-headline-count data-value">{detections.length}</span>
        {detections.length === 1 ? " match" : " matches"}
        {data.query ? <span className="detection-headline-query"> for &ldquo;{String(data.query)}&rdquo;</span> : null}
      </p>
      {detections.length > 0 && <DetectionTally detections={detections} />}
      {sourceImageUrl ? (
        <GroundingOverlay imageUrl={evidenceImageUrl(sourceImageUrl)} detections={detections} />
      ) : (
        <p className="tool-card-note">No source image available to overlay detections on.</p>
      )}
      {detections.length > 0 && <DetectionList detections={detections} />}
    </div>
  );
}

function SceneDescriptionCard({ data, sourceImageUrl }: { data: Record<string, unknown>; sourceImageUrl: string | null }) {
  const detections = (data.detections ?? []) as Detection[];
  const scanned = ((data.scanned_categories ?? []) as string[]).join(", ");
  return (
    <div className="tool-card-body">
      <p className="detection-headline">
        <span className="detection-headline-count data-value">{detections.length}</span>
        {detections.length === 1 ? " object found" : " objects found"}
      </p>
      {detections.length > 0 && <DetectionTally detections={detections} />}
      {sourceImageUrl && detections.length > 0 && (
        <GroundingOverlay imageUrl={evidenceImageUrl(sourceImageUrl)} detections={detections} />
      )}
      <p className="tool-card-note">
        The scan only checks for {scanned || "a few object types"}; anything else in the scene is not reported here.
      </p>
      {detections.length > 0 && <DetectionList detections={detections} />}
    </div>
  );
}

function FusionCard({ data, evidenceUrl }: { data: Record<string, unknown>; evidenceUrl: string | null }) {
  const agree = data.water_agreement === "agree";
  const waterSar = Number(data.water_fraction_sar ?? 0);
  const waterOptical = Number(data.water_fraction_optical ?? 0);
  const builtup = Number(data.builtup_fraction_sar ?? 0);
  return (
    <div className="tool-card-body">
      <span className={`agree-badge ${agree ? "agree" : "disagree"}`}>
        {agree ? "SAR and optical agree" : "SAR and optical disagree"}
      </span>
      {evidenceUrl && <EvidenceImage url={evidenceUrl} />}
      <DetailToggle>
        <Bar label="Water (SAR)" value={waterSar} percent />
        <Bar label="Water (optical)" value={waterOptical} percent />
        <Bar label="Built-up (SAR-only, coarser read)" value={builtup} percent />
      </DetailToggle>
    </div>
  );
}

interface ClassChange {
  class: string;
  before: number;
  after: number;
  net: number;
  direction: "increased" | "decreased" | "unchanged";
}

interface Transition {
  from: string;
  to: string;
  fraction: number;
}

function signedPct(fraction: number): string {
  return `${fraction >= 0 ? "+" : "\u2212"}${Math.abs(fraction * 100).toFixed(1)}%`;
}

// One class's net area change as a diverging bar around a center line -- longest bar in the set
// fills half the track, so the rows read against each other rather than against an absolute scale.
function NetChangeRow({ change, scale }: { change: ClassChange; scale: number }) {
  const width = Math.min(50, (Math.abs(change.net) / scale) * 50);
  const up = change.net >= 0;
  return (
    <div className="delta-row">
      <span className="delta-label">{change.class}</span>
      <div className="delta-track" aria-hidden="true">
        <div
          className={`delta-fill ${up ? "delta-up" : "delta-down"}`}
          style={up ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
        />
      </div>
      <span className="data-value delta-value">{signedPct(change.net)}</span>
    </div>
  );
}

function ChangeDetectionCard({ data, evidenceUrl }: { data: Record<string, unknown>; evidenceUrl: string | null }) {
  const fraction = Number(data.change_fraction ?? 0);
  const bbox = data.largest_region_bbox as [number, number, number, number] | null;
  // Stage 2 (semantic model) adds per-class net change + transitions; Stage 1 (pixel differencing)
  // returns only change_fraction/bbox, and renders exactly as it did before.
  const classChanges = (data.class_changes ?? []) as ClassChange[];
  const transitions = (data.transitions ?? []) as Transition[];
  const semantic = data.method === "semantic" && classChanges.length > 0;
  const buildings = classChanges.find((c) => c.class === "buildings");
  const moved = classChanges.filter((c) => c.direction !== "unchanged").sort((a, b) => Math.abs(b.net) - Math.abs(a.net));
  const scale = Math.max(0.01, ...classChanges.map((c) => Math.abs(c.net)));

  return (
    <div className="tool-card-body">
      {semantic && buildings && (
        <span className={`trend-badge trend-${buildings.direction}`}>
          Built-up (buildings):{" "}
          {buildings.direction === "unchanged" ? "unchanged" : `${buildings.direction} ${signedPct(buildings.net)}`}
        </span>
      )}
      <Bar label="Changed area" value={fraction} percent />
      {evidenceUrl && <EvidenceImage url={evidenceUrl} wide />}
      {(semantic || bbox) && (
        <DetailToggle>
          {semantic && (
            <>
              <div className="detail-heading">Net area change by class (share of the whole scene)</div>
              {moved.length > 0 ? (
                moved.map((c) => <NetChangeRow key={c.class} change={c} scale={scale} />)
              ) : (
                <p className="tool-card-note">No class gained or lost a meaningful share of the scene.</p>
              )}
              {transitions.length > 0 && (
                <>
                  <div className="detail-heading">Largest transitions</div>
                  <ul className="detection-list">
                    {transitions.slice(0, 5).map((t, i) => (
                      <li key={i}>
                        <span>
                          {t.from} → {t.to}
                        </span>
                        <span className="data-value">{(t.fraction * 100).toFixed(1)}%</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
          {bbox && (
            <dl className="kv-grid">
              <dt>Region bbox</dt>
              <dd className="data-value">[{bbox.map((n) => Math.round(n)).join(", ")}]</dd>
            </dl>
          )}
        </DetailToggle>
      )}
    </div>
  );
}

function WaterSegmentationCard({ data, evidenceUrl }: { data: Record<string, unknown>; evidenceUrl: string | null }) {
  return (
    <div className="tool-card-body">
      <Bar label="Water fraction" value={Number(data.water_fraction ?? 0)} percent />
      {evidenceUrl && <EvidenceImage url={evidenceUrl} />}
    </div>
  );
}

function VqaCard({ data }: { data: Record<string, unknown> }) {
  return (
    <div className="tool-card-body">
      <p className="vqa-question">“{String(data.question ?? "")}”</p>
      <p className="vqa-answer">{String(data.answer ?? "")}</p>
    </div>
  );
}

function FallbackCard({ result }: { result: ToolResultOut }) {
  return (
    <div className="tool-card-body">
      <p className="tool-card-note">{result.text_summary}</p>
      {Object.keys(result.structured_data).length > 0 && (
        <DetailToggle>
          <dl className="kv-grid">
            {Object.entries(result.structured_data).map(([k, v]) => (
              <Fragment key={k}>
                <dt>{k.replace(/_/g, " ")}</dt>
                <dd className="data-value">{typeof v === "object" ? JSON.stringify(v) : String(v)}</dd>
              </Fragment>
            ))}
          </dl>
        </DetailToggle>
      )}
    </div>
  );
}

export function ToolResultCard({ result }: { result: ToolResultOut }) {
  const data = result.structured_data;
  let body: ReactNode;
  switch (result.tool_name) {
    case "groundwater_potential":
      body = <GroundwaterCard data={data} evidenceUrl={result.evidence_image_url} />;
      break;
    case "wildfire_detection":
      body = <WildfireCard data={data} />;
      break;
    case "text_guided_grounding":
      body = <GroundingCard data={data} sourceImageUrl={result.source_image_url} />;
      break;
    case "scene_description":
      body = <SceneDescriptionCard data={data} sourceImageUrl={result.source_image_url} />;
      break;
    case "optical_sar_fusion":
      body = <FusionCard data={data} evidenceUrl={result.evidence_image_url} />;
      break;
    case "change_detection":
      body = <ChangeDetectionCard data={data} evidenceUrl={result.evidence_image_url} />;
      break;
    case "water_body_segmentation":
      body = <WaterSegmentationCard data={data} evidenceUrl={result.evidence_image_url} />;
      break;
    case "visual_question_answering":
      body = <VqaCard data={data} />;
      break;
    default:
      body = <FallbackCard result={result} />;
  }

  return (
    <div className="tool-card">
      <div className="tool-card-header">
        <span className="tool-card-name">{TOOL_LABEL[result.tool_name] ?? result.tool_name}</span>
        <span className="data-value tool-card-confidence">{result.confidence.toFixed(2)}</span>
      </div>
      {body}
    </div>
  );
}
