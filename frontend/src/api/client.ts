// Mirrors backend/app/schemas/models.py exactly -- keep these two in sync by hand for now
// (Phase 0 scope; codegen from the OpenAPI schema is a reasonable later upgrade, not needed yet).

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export interface GeoMetadata {
  source: "geotiff" | "exif" | "map_capture";
  center_lat: number;
  center_lon: number;
  bounds_wgs84: [number, number, number, number] | null;
  crs: string | null;
  band_count: number | null;
  acquisition_datetime: string | null;
}

export interface UploadedImageInfo {
  filename: string;
  stored_path: string;
  format: string;
  width: number;
  height: number;
  modality_guess: string;
  geo: GeoMetadata | null;
}

export interface UploadResponse {
  input_id: string;
  images: UploadedImageInfo[];
  warnings: string[];
  errors: string[]; // files that failed validation and were NOT saved (e.g. unsupported format)
}

export interface ToolUsage {
  name: string;
  params: Record<string, unknown>;
  checkpoint_id: string | null;
  // Which agentic-loop round called this tool (backend/app/orchestrator/controller.py's
  // MAX_AGENT_ROUNDS loop) -- 1 unless a later round's call was informed by an earlier round's own
  // results. Not rendered specially anywhere in the UI yet; kept honest in the type for when it is.
  round: number;
}

export interface ExecutionTrace {
  selected_task: string;
  tools_used: ToolUsage[];
  input_summary: string;
  confidence: number;
  confidence_bucket: "High" | "Medium" | "Low";
  warnings: string[];
  timestamp: string;
}

export interface ToolResultOut {
  tool_name: string;
  text_summary: string;
  structured_data: Record<string, unknown>;
  confidence: number;
  evidence_image_url: string | null;
  source_image_url: string | null;
}

export interface QueryResponse {
  query_id: string;
  answer_text: string;
  confidence: number;
  confidence_bucket: "High" | "Medium" | "Low";
  evidence_image_urls: string[];
  tool_results: ToolResultOut[];
  execution_trace: ExecutionTrace;
}

export interface ToolSpecOut {
  name: string;
  description: string;
  parameters_schema: {
    type: string;
    properties: Record<string, { type: string; description?: string }>;
    required?: string[];
  };
  min_images: number;
  max_images: number;
  compatible_modalities: string[];
  uses_images: boolean;
  requires_location: boolean;
  checkpoint_id: string | null;
}

export interface ForcedToolCall {
  tool_name: string;
  arguments: Record<string, unknown>;
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}

// `response.json()` is typed `any` under the hood -- nothing previously caught a backend field
// rename before it silently rendered as `undefined` somewhere deep in a component. Not a full
// schema validation (no new dependency for this) -- just the top-level keys each interface
// declares, which is exactly what a rename/typo would actually break.
function assertShape(body: unknown, requiredKeys: string[], context: string): void {
  if (typeof body !== "object" || body === null) {
    throw new Error(`${context}: expected a JSON object in the response, got ${typeof body}`);
  }
  const missing = requiredKeys.filter((key) => !(key in (body as Record<string, unknown>)));
  if (missing.length > 0) {
    throw new Error(`${context}: response is missing expected field(s) ${missing.join(", ")} -- backend/frontend may be out of sync`);
  }
}

const UPLOAD_RESPONSE_KEYS = ["input_id", "images", "warnings", "errors"];
const QUERY_RESPONSE_KEYS = [
  "query_id",
  "answer_text",
  "confidence",
  "confidence_bucket",
  "evidence_image_urls",
  "tool_results",
  "execution_trace",
];

export async function uploadImages(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);

  const response = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(`Upload failed: ${await parseErrorDetail(response)}`);
  const body = await response.json();
  assertShape(body, UPLOAD_RESPONSE_KEYS, "Upload response");
  return body;
}

export interface AreaBounds {
  min_lat: number;
  min_lon: number;
  max_lat: number;
  max_lon: number;
}

export async function captureArea(bounds: AreaBounds): Promise<UploadResponse> {
  const response = await fetch(`${API_BASE}/api/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bounds),
  });
  if (!response.ok) throw new Error(`Capture failed: ${await parseErrorDetail(response)}`);
  const body = await response.json();
  assertShape(body, UPLOAD_RESPONSE_KEYS, "Capture response");
  return body;
}

export interface LocationIn {
  lat: number;
  lon: number;
}

export async function runQuery(
  inputId: string | null,
  queryText: string,
  location?: LocationIn | null,
  forcedTools?: ForcedToolCall[] | null,
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input_id: inputId,
      query_text: queryText,
      location: location ?? null,
      forced_tools: forcedTools ?? null,
    }),
  });
  if (!response.ok) throw new Error(`Query failed: ${await parseErrorDetail(response)}`);
  const body = await response.json();
  assertShape(body, QUERY_RESPONSE_KEYS, "Query response");
  return body;
}

// Live registry metadata for every specialist -- drives the homepage capabilities gallery and the
// composer's manual "Advanced" tool picker from one source that can't drift from what actually runs.
export async function listTools(): Promise<ToolSpecOut[]> {
  const response = await fetch(`${API_BASE}/api/tools`);
  if (!response.ok) throw new Error(`Failed to load tool list: ${await parseErrorDetail(response)}`);
  return response.json();
}

export function evidenceImageUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function reportUrl(queryId: string): string {
  return `${API_BASE}/api/report/${queryId}`;
}
