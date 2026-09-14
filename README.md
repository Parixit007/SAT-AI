# SatQuery AI

**An agentic vision-language assistant for remote-sensing imagery.** Ask a natural-language
question about a satellite image, a bi-temporal pair, or an optical+SAR pair — an LLM-driven
controller routes the query to the right specialist model(s), executes them, and returns an
evidence-grounded answer with a full, auditable execution trace.

This isn't a single VLM answering everything from vibes. Every answer traces back to a named tool,
a checkpoint (or an explicit "no model, deterministic computation"), and a confidence score whose
meaning is documented per tool — because "confidence: 0.83" means something different coming from
a segmentation model's pixel probabilities than from a GIS overlay's data-completeness count, and
papering over that difference would make the whole system less trustworthy, not more.

Built for a remote-sensing AI assignment modeled on an ISRO/SAC-style evaluation — see
[`problem_statement.txt`](problem_statement.txt) for the full spec this was built against.

## What it can do

| Capability | Status | How |
|---|---|---|
| **Visual question answering** | ✅ Working | PaliGemma fine-tuned on RSVQA-LR |
| **Text-guided region grounding** | ✅ Working | Grounding DINO fine-tuned on DIOR-RSVG |
| **Bi-temporal change analysis** | ✅ Working (Stage 1) | Adaptive pixel-differencing; semantic segmentation model queued |
| **Optical–SAR cross-modal fusion** | ✅ Working (Stage 1) | SAR-backscatter physics, cross-checked against the optical water read |
| **Groundwater potential** ("should I dig a well here?") | ✅ Working | Google Earth Engine, AHP-weighted GIS overlay — no model, fully deterministic |
| **Agentic orchestration** | ✅ Working | LLM picks tools; everything else (validation, execution, trace) is deterministic Python |

All five mandatory capabilities from the spec are implemented end to end. "Stage 1" tools are
intentionally training-free so they work today; each has a Stage 2 fine-tuned upgrade path already
scoped (see [`CLAUDE.md`](CLAUDE.md) for the detailed roadmap).

## How a query actually gets answered

```
query + image(s)/location
        │
        ▼
 input validation ──────► format/count/dimension checks, geo-metadata extraction
        │                  (no LLM call — deterministic Python)
        ▼
 LLM tool selection ────► Gemini or Groq picks which specialist(s) to call,
        │                  given the query + a text summary of the input
        │                  (never given raw pixels — this measures tool-calling
        │                  quality, not vision)
        ▼
 compatibility check ───► does the input actually satisfy what the chosen tool
        │                  needs (image count, modality, location)? One retry
        │                  on mismatch, then a recorded skip — never a crash.
        ▼
 execution ─────────────► the specialist runs; a failure here (missing
        │                  dependency, gated model, upstream API down) becomes
        │                  a trace warning, not a 500
        ▼
 execution trace ───────► selected task, tool(s) used, checkpoint id(s),
                           confidence, warnings, timestamp — built from what
                           actually ran, never from the LLM's own narration
```

The LLM makes exactly one decision: which tool(s) to call. It never sees raw pixels, never phrases
the final answer, and never writes anything into the trace directly — that separation is what
makes the trace auditable rather than just an LLM's word for it.

## Quick start

**Backend** (FastAPI):
```bash
pip install -r backend/requirements.txt
cp .env.example .env   # fill in at least one of GEMINI_API_KEY / GROQ_API_KEY
cd backend && uvicorn app.main:app --reload --port 8000
```

**Frontend** (React + Vite):
```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Without an LLM key, `/api/query` returns a clean 503 — not a crash. The groundwater tool similarly
needs `GEE_PROJECT_ID`/`GEE_SERVICE_ACCOUNT_EMAIL`/`GEE_SERVICE_ACCOUNT_KEY_FILE`; VQA needs
`HF_TOKEN` (the checkpoint is gated on Hugging Face). Every other tool works with no extra setup.

Run the test suite:
```bash
cd backend && pytest -q   # 68 passed, 1 skipped (the skip needs groundingdino installed locally)
```

## The interface

A full-bleed map is the location picker — click it, or upload a georeferenced image and it drops
the pin itself. A fixed side panel holds the query box, example prompts, image upload, and results:
answer text, a confidence badge, an evidence image where the tool produces one (a mask overlay, a
bounding box, a before/after composite), and a collapsible execution trace for the full audit
detail. Every query result is downloadable as a plain-text report.

## Specialist models

| Tool | Model | Training data |
|---|---|---|
| `visual_question_answering` | PaliGemma-3B | RSVQA-LR (Google's own fine-tune; our own LoRA fine-tune notebook included) |
| `text_guided_grounding` | Grounding DINO (Swin-T) | DIOR-RSVG + VRSBench referring expressions |
| `water_body_segmentation` | U-Net (ResNet-34 encoder) | Satellite Images of Water Bodies (Kaggle) |
| `change_detection` | Otsu-adaptive pixel differencing | none needed (Stage 1); SECOND-CC queued for Stage 2 |
| `optical_sar_fusion` | Recursive-Otsu SAR backscatter analysis | none needed (Stage 1); TUM SEN1-2 queued for Stage 2 |
| `groundwater_potential` | AHP-weighted GIS overlay (rainfall, TWI, land cover, distance-to-water) | not a trained model — pure Earth Engine computation |

Every image-based tool follows the same shape: a `*Tool` class in `models/<name>/` with lazy
imports and one inference method, plus a separate `draw_*()` function for evidence-image
rendering — inference never touches the image directly, drawing is always an explicit, skippable
step. See [`CLAUDE.md`](CLAUDE.md) for the full architectural writeup, including *why* each design
decision was made, not just what it is.

## Training on Kaggle

The local dev machine has no NVIDIA GPU, so every fine-tune happens on Kaggle. `notebooks/`
currently has four notebooks — grounding (original + a v2 continued run adding VRSBench referring
data), water segmentation, and VQA LoRA fine-tuning — with change detection and fusion's Stage 2
notebooks scoped but not yet written. Each follows the same 10-section structure (setup → data →
conversion → config → train → eval → export), so a new one is easy to read once you've seen the
first. Every dataset choice, box-coordinate convention, and split methodology in these notebooks
was verified against the actual downloaded data or the original authors' source before being
written — not assumed.

## Tech stack

**Backend:** FastAPI · PyTorch · `segmentation-models-pytorch` · Transformers · `rasterio` ·
Google Earth Engine · Gemini / Groq (swappable LLM tool-calling backend)
**Frontend:** React · TypeScript · Vite · Leaflet
**Data:** DIOR-RSVG, VRSBench, RSVQA-LR, CDVQA, BigEarthNet, SECOND-CC, TUM SEN1-2 — see
`data/scripts/` for the download scripts and `CLAUDE.md` for licensing notes per dataset.

## Project structure

```
backend/app/            FastAPI app + orchestrator
  orchestrator/          input validation, LLM tool-calling, execution trace
  specialists/            thin adapters wiring models/ into the tool registry
  gee/                    Google Earth Engine layers + groundwater scoring
backend/tests/           pytest suite — orchestrator, API, and per-specialist tests
frontend/src/            React + Vite + TS single-page app
models/                  standalone specialist wrapper scripts (grounding, VQA, water, change, fusion)
notebooks/                Kaggle fine-tuning notebooks (grounding, water segmentation, VQA)
data/scripts/             dataset download scripts
```

## Status

Actively developed. All mandatory capabilities are implemented and tested (68 tests passing); the
current focus is Stage 2 fine-tuning for change detection and fusion, plus ongoing hardening. See
[`CLAUDE.md`](CLAUDE.md) for the full current-state writeup and phased roadmap.
