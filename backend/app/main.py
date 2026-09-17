from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import routes_query, routes_reports, routes_tools, routes_upload
from app.config import EVIDENCE_DIR, UPLOADS_DIR, settings

app = FastAPI(title="SatQuery AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")
# Serves the original uploaded images (not just derived evidence renders) -- needed so the
# frontend can draw an interactive overlay (e.g. grounding's boxes) on top of the actual source
# image, since box coordinates are in that image's own pixel space (see ToolResultOut.source_image_url).
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

app.include_router(routes_upload.router, prefix="/api", tags=["upload"])
app.include_router(routes_query.router, prefix="/api", tags=["query"])
app.include_router(routes_reports.router, prefix="/api", tags=["reports"])
app.include_router(routes_tools.router, prefix="/api", tags=["tools"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
