"""Lazy Earth Engine initialization -- same lazy-singleton shape as specialists/*_adapter.py's
_get_tool(). Registering a GEE-backed tool (cheap, metadata-only) always works even before GEE is
configured; only actually running one calls ensure_initialized() and can raise."""

from app.concurrency import serialize_first_call
from app.config import settings

_initialized = False


@serialize_first_call
def ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return

    import ee

    if not (settings.gee_service_account_email and settings.gee_service_account_key_file and settings.gee_project_id):
        raise RuntimeError(
            "Google Earth Engine is not configured -- set GEE_PROJECT_ID, GEE_SERVICE_ACCOUNT_EMAIL, "
            "and GEE_SERVICE_ACCOUNT_KEY_FILE in .env (see .env.example)."
        )

    credentials = ee.ServiceAccountCredentials(
        settings.gee_service_account_email, settings.gee_service_account_key_file
    )
    ee.Initialize(credentials, project=settings.gee_project_id)
    _initialized = True
