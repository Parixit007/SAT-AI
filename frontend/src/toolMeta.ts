// Small hand-maintained UI copy per specialist, keyed by the exact ToolSpec.name values registered
// in backend/app/specialists/*_adapter.py -- same key space ResultsView.tsx's CONFIDENCE_SEMANTICS
// already established. Structural facts (what tools exist, their params, compatibility) come from
// the live GET /api/tools registry instead (see api/client.ts's listTools()) so they can't drift;
// this file only holds presentation copy the backend has no reason to own (an icon, a short label,
// one good example query) -- shared by QueryEntry, ToolResultCard, the homepage gallery, and the
// composer's Advanced panel so none of them can drift from each other either.

export const TOOL_ICON: Record<string, string> = {
  groundwater_potential: "💧",
  wildfire_detection: "🔥",
  text_guided_grounding: "🎯",
  scene_description: "🗺️",
  optical_sar_fusion: "🛰️",
  change_detection: "🔀",
  water_body_segmentation: "🌊",
  visual_question_answering: "❓",
  land_cover_analysis: "🏘️",
};

export const TOOL_LABEL: Record<string, string> = {
  groundwater_potential: "Groundwater potential",
  wildfire_detection: "Wildfire detection",
  text_guided_grounding: "Object detection & counting",
  scene_description: "Scene description",
  optical_sar_fusion: "Optical–SAR fusion",
  change_detection: "Change detection",
  water_body_segmentation: "Water segmentation",
  visual_question_answering: "Visual Q&A",
  land_cover_analysis: "Land cover & buildings",
};

// From problem_statement.txt's "Representative Queries", plus one each for groundwater/wildfire --
// carried over from the composer's old inline EXAMPLES array.
export const EXAMPLE_PROMPT: Record<string, string> = {
  water_body_segmentation: "Highlight the water body referred to in the query.",
  groundwater_potential: "Should I dig a well/tubewell at this location?",
  visual_question_answering: "Is there a road in this image?",
  scene_description: "Describe the land-cover and major objects visible in this image.",
  change_detection: "Has the built-up area increased, decreased, or remained unchanged between these two dates?",
  optical_sar_fusion: "Use the optical and SAR images together to identify built-up and water-covered regions.",
  text_guided_grounding: "How many airplanes are in this image?",
  wildfire_detection: "Is there any active wildfire near this location?",
  land_cover_analysis: "How much of this area is buildings, roads and vegetation, and how many buildings are there?",
};
