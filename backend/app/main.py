from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import routes_query, routes_reports, routes_upload
from app.config import EVIDENCE_DIR, settings

app = FastAPI(title="SatQuery AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

app.include_router(routes_upload.router, prefix="/api", tags=["upload"])
app.include_router(routes_query.router, prefix="/api", tags=["query"])
app.include_router(routes_reports.router, prefix="/api", tags=["reports"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
