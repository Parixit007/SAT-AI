// Mirrors backend/app/schemas/models.py exactly -- keep these two in sync by hand for now
// (Phase 0 scope; codegen from the OpenAPI schema is a reasonable later upgrade, not needed yet).

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export interface GeoMetadata {
  source: "geotiff" | "exif";
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

export interface QueryResponse {
  query_id: string;
  answer_text: string;
  confidence: number;
  confidence_bucket: "High" | "Medium" | "Low";
  evidence_image_urls: string[];
  execution_trace: ExecutionTrace;
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function uploadImages(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);

  const response = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(`Upload failed: ${await parseErrorDetail(response)}`);
  return response.json();
}

export interface LocationIn {
  lat: number;
  lon: number;
}

export async function runQuery(
  inputId: string | null,
  queryText: string,
  location?: LocationIn | null,
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input_id: inputId, query_text: queryText, location: location ?? null }),
  });
  if (!response.ok) throw new Error(`Query failed: ${await parseErrorDetail(response)}`);
  return response.json();
}

export function evidenceImageUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function reportUrl(queryId: string): string {
  return `${API_BASE}/api/report/${queryId}`;
}
