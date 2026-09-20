import { useState } from "react";

export interface Detection {
  phrase: string;
  bbox_xyxy: [number, number, number, number];
  score: number;
}

const BOX_COLORS = ["#cf4433", "#2f7fc4", "#1f9968", "#b3781e", "#7c5fd6"];

// Above this many boxes a text label on each one just covers the image (a count of 40 airplanes is
// 40 labels): draw outlines only and let the legend above carry the per-phrase tally instead.
const LABEL_LIMIT = 12;

export function phraseColors(detections: Detection[]): Map<string, string> {
  const colors = new Map<string, string>();
  for (const d of detections) {
    if (!colors.has(d.phrase)) colors.set(d.phrase, BOX_COLORS[colors.size % BOX_COLORS.length]);
  }
  return colors;
}

// grounding_tool.py's ground() returns bbox_xyxy in the ORIGINAL uploaded image's absolute pixel
// coordinates (see models/grounding/grounding_tool.py) -- so boxes are drawn over that source
// image, not the pre-composited evidence JPEG, and positioned as percentages of the image's own
// natural size. Percentages (not px) mean the overlay stays aligned at any rendered display size
// with no resize listener needed.
export function GroundingOverlay({ imageUrl, detections }: { imageUrl: string; detections: Detection[] }) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const colors = phraseColors(detections);
  const labelled = detections.length <= LABEL_LIMIT;

  return (
    <div className="grounding-overlay">
      <img
        src={imageUrl}
        alt="Source image with detected regions"
        className="grounding-overlay-image"
        onLoad={(e) => {
          const img = e.currentTarget;
          setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
        }}
      />
      {naturalSize &&
        detections.map((d, i) => {
          const [x1, y1, x2, y2] = d.bbox_xyxy;
          const color = colors.get(d.phrase) ?? BOX_COLORS[0];
          return (
            <div
              key={i}
              className={`grounding-box ${labelled ? "" : "grounding-box-thin"}`}
              style={{
                left: `${(x1 / naturalSize.w) * 100}%`,
                top: `${(y1 / naturalSize.h) * 100}%`,
                width: `${((x2 - x1) / naturalSize.w) * 100}%`,
                height: `${((y2 - y1) / naturalSize.h) * 100}%`,
                borderColor: color,
                background: labelled ? `${color}22` : "transparent",
              }}
            >
              {labelled && (
                <span className="grounding-box-label" style={{ background: color }}>
                  {d.phrase} <span className="grounding-box-score">{d.score.toFixed(2)}</span>
                </span>
              )}
            </div>
          );
        })}
    </div>
  );
}

// "airplane x 39" chips, coloured like the boxes they count.
export function DetectionTally({ detections }: { detections: Detection[] }) {
  const colors = phraseColors(detections);
  const counts = new Map<string, number>();
  for (const d of detections) counts.set(d.phrase, (counts.get(d.phrase) ?? 0) + 1);
  return (
    <div className="detection-tally">
      {[...counts.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([phrase, n]) => (
          <span key={phrase} className="detection-chip" style={{ borderColor: colors.get(phrase), color: colors.get(phrase) }}>
            <span className="detection-chip-count data-value">{n}</span> {phrase}
          </span>
        ))}
    </div>
  );
}
