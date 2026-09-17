import { useState } from "react";

interface Detection {
  phrase: string;
  bbox_xyxy: [number, number, number, number];
  score: number;
}

const BOX_COLORS = ["#2f7fc4", "#1f9968", "#b3781e", "#cf4433", "#7c5fd6"];

// grounding_tool.py's ground() returns bbox_xyxy in the ORIGINAL uploaded image's absolute pixel
// coordinates (see models/grounding/grounding_tool.py) -- so boxes are drawn over that source
// image, not the pre-composited evidence JPEG, and positioned as percentages of the image's own
// natural size. Percentages (not px) mean the overlay stays aligned at any rendered display size
// with no resize listener needed.
export function GroundingOverlay({ imageUrl, detections }: { imageUrl: string; detections: Detection[] }) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);

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
          const color = BOX_COLORS[i % BOX_COLORS.length];
          return (
            <div
              key={i}
              className="grounding-box"
              style={{
                left: `${(x1 / naturalSize.w) * 100}%`,
                top: `${(y1 / naturalSize.h) * 100}%`,
                width: `${((x2 - x1) / naturalSize.w) * 100}%`,
                height: `${((y2 - y1) / naturalSize.h) * 100}%`,
                borderColor: color,
                background: `${color}22`,
              }}
            >
              <span className="grounding-box-label" style={{ background: color }}>
                {d.phrase} <span className="grounding-box-score">{d.score.toFixed(2)}</span>
              </span>
            </div>
          );
        })}
    </div>
  );
}
