# SatQuery AI

**An agentic vision-language assistant for remote-sensing imagery.** Ask a natural-language
question about a satellite image, a bi-temporal pair, an optical+SAR pair or just a location — an
LLM-driven controller routes the query to the right specialist model(s), runs them, and returns an
evidence-grounded answer with an auditable execution trace.

This is not one VLM answering everything from vibes. Every answer traces back to a named tool, a
checkpoint (or an explicit "no model, deterministic computation"), and a confidence score whose
meaning is documented per tool — "confidence 0.83" means something different from a segmentation
model's pixel probabilities than from a GIS overlay's data-completeness count, and papering over that
would make the system less trustworthy, not more.

Built for a remote-sensing AI assignment modelled on an ISRO/SAC-style evaluation — see
[`problem_statement.txt`](problem_statement.txt) for the spec.

## What it can do

| Capability | Tool | Status |
|---|---|---|
| Visual question answering | `visual_question_answering` | Working — PaliGemma-3B (Google's RSVQA-LR fine-tune); one- or two-word answers |
| Find and **count** named objects | `text_guided_grounding` | Working — Grounding DINO fine-tuned on DIOR-RSVG + VRSBench + DOTA; vehicles counted on native-resolution tiles (a lower bound) |
| "Describe this image" | `scene_description` | Working — three sources: a remote-sensing captioner, land-cover measurements and an object scan |
| **Buildings, roads, vegetation, water** — shares, building count, roof colours | `land_cover_analysis` | Working — U-Net trained on OpenEarthMap (validation mIoU 0.63) |
| Water bodies | `water_body_segmentation` | Working — U-Net (validation IoU 0.79) |
| Bi-temporal change ("has the built-up area increased?") | `change_detection` | Working — Siamese semantic-change net trained on SECOND-CC; pixel-differencing fallback |
| Optical–SAR analysis | `optical_sar_fusion` | Stage 1 — SAR-backscatter physics reconciled with the optical water read (training-free) |
| Groundwater potential ("should I dig a well here?") | `groundwater_potential` | Working — Google Earth Engine, weighted GIS overlay (deterministic) |
| Active wildfire nearby | `wildfire_detection` | Working — NASA FIRMS via Earth Engine |
| Agentic orchestration | controller | Working — LLM picks tools; everything else is deterministic Python |

Bounds worth knowing (measured, and written up in [`CLAUDE.md`](CLAUDE.md)): the captioner was right
in gist for 19 of 29 real scenes, partly right for 8 and wrong for 2, and its object counts are not
trusted; building counts under-count dense blocks because touching buildings merge; the land-cover
model mislabels shadowed streets, dune shadows and some desert terrain; the optical–SAR tool and the
VQA model are not yet benchmarked or fine-tuned by us; there is no end-to-end benchmark harness yet.

## How a query is answered

```
query + image(s) / location
        │
        ▼
 1 validate ─────► format, count and size checks; real geo-metadata from GeoTIFF / EXIF
        │           (deterministic Python — no LLM)
        ▼
 2 route ────────► Gemini or Groq picks the tool(s) from the query and a TEXT summary of the
        │           input — never pixels. The UI can bypass this with a manual tool picker.
        ▼
 3 check ────────► does the input satisfy the tool (image count, modality, location)?
        │           One retry, then a recorded skip — never a crash.
        ▼
 4 execute ──────► up to 3 tools in parallel; a failing tool becomes a trace warning, not a 500
        │
        ▼
 5 compose ──────► per-tool confidence combined; the LLM writes the answer ONLY from the tools'
        │           outputs, and every multi-digit number in it is checked against the evidence
        │           (one retry, then the plain deterministic text is kept)
        ▼
 6 trace ────────► selected task, tools, checkpoint ids, confidence, warnings, timestamp — built
                    from what actually ran, never from the LLM's own narration
```

The LLM makes exactly two decisions: which tools to call, and how to phrase what they found. It never
sees raw pixels and never writes into the trace, so uploaded imagery stays on the machine; the only
outbound calls are text to the LLM provider, map-tile requests to Esri and coordinates to Earth Engine.

## Quick start

**Backend** (FastAPI):
```bash
pip install -r backend/requirements.txt
cp .env.example .env   # set at least one of GEMINI_API_KEY / GROQ_API_KEY
cd backend && uvicorn app.main:app --reload --port 8000
```

**Frontend** (React + Vite):
```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

**Tests:**
```bash
cd backend && pytest -q          # 244 passed, 1 skipped
cd frontend && npm run build     # type-checks, then builds
```

**Model checkpoints are not in git.** Each specialist loads a checkpoint from
`models/<name>/checkpoints/` (gitignored) that the matching notebook in `notebooks/` produces on
Kaggle. A specialist whose checkpoint is missing fails with a clear message and the rest keep working;
`scene_description` simply uses whichever of its sources are installed.

| Needs | For |
|---|---|
| `GEMINI_API_KEY` or `GROQ_API_KEY` | routing and answer phrasing (without one `/api/query` returns a clean 503) |
| a one-time `git clone` of Open-GroundingDino + a few pip packages (see `models/grounding/grounding_tool.py`) | grounding and counting |
| `HF_TOKEN` (the PaliGemma checkpoint is gated) | visual question answering |
| `GEE_PROJECT_ID`, `GEE_SERVICE_ACCOUNT_EMAIL`, `GEE_SERVICE_ACCOUNT_KEY_FILE` | groundwater and wildfire |
| `ENABLE_CAPTIONER`, `ENABLE_LANDCOVER` (both default true) | use the captioner / land-cover model inside "describe this image" when installed |

## The interface

A report-panel layout, not a chat log: each query is a full-width entry with the answer, a confidence
badge, and one **evidence card per tool** — bar charts, colour-coded segmentation overlays, detection
boxes drawn over the original image, before/after composites — plus a collapsible execution trace and
a text report download. Around it: a capabilities gallery on the home screen (click a card to fill in
an example query), an **Advanced** panel to pick tools manually with parameter forms generated from
each tool's JSON schema, and a **Leaflet map** with real Esri imagery where you can drop a pin or draw
an area and capture it as a georeferenced image that flows through the same path as an upload.

## Specialist models

| Tool | Model | Trained on | Measured |
|---|---|---|---|
| `text_guided_grounding` | Grounding DINO (Swin-T) | DIOR-RSVG + VRSBench + DOTA | Acc@0.5 0.827, mIoU 0.746 (DIOR-RSVG test subset) |
| `scene_description` (captioner) | SmolVLM-500M + LoRA | VRSBench + NWPU-Captions | BLEU-4 0.111, CIDEr 0.244 on 1,000 VRSBench images; real-scene reading above |
| `land_cover_analysis` | U-Net (ResNet-34) | OpenEarthMap | mIoU 0.633; building IoU 0.771; building count correlation 0.88 |
| `water_body_segmentation` | U-Net (ResNet-34) | Satellite Images of Water Bodies | validation IoU 0.7945 |
| `change_detection` | Siamese semantic-change U-Net | SECOND-CC | Score 0.376; buildings direction 84% (guessing "unchanged" gets 40%) |
| `visual_question_answering` | PaliGemma-3B | RSVQA-LR (Google's fine-tune) | not evaluated here — no held-out RSVQA-LR answers exist locally |
| `optical_sar_fusion` | recursive Otsu on SAR + the water model | none (Stage 1) | not benchmarked |
| `groundwater_potential`, `wildfire_detection` | deterministic Earth Engine computations | none | not validated against ground truth |

Every image-based tool has the same shape: a `*Tool` class in `models/<name>/` with lazy imports and one
inference method, plus a separate `draw_*()` for the evidence image. Adding a specialist is one adapter
module in `backend/app/specialists/` and one line in `build_default_registry()`. [`CLAUDE.md`](CLAUDE.md)
records why each design decision was made, including what was measured and what went wrong.

## Training on Kaggle

The dev machine has no NVIDIA GPU, so every fine-tune runs on Kaggle (`notebooks/`, ten notebooks):
grounding (original, a superseded continuation, a DOTA-augmented v3 and its eval-only companion), water
segmentation, the semantic change model, the captioner (VRSBench, then a scene-mix round with
NWPU-Captions), the land-cover segmenter, and a VQA LoRA notebook that is written but not yet run. Each
was run end to end locally on real data in a smoke mode before any GPU time was spent, and every dataset
convention (coordinate systems, label encodings, split leakage) was checked against the real files —
several real bugs were found that way, and are recorded in `CLAUDE.md`.

## Tech stack

**Backend:** FastAPI · PyTorch · `segmentation-models-pytorch` · Transformers · `rasterio` ·
Google Earth Engine · Gemini / Groq (swappable tool-calling backend)
**Frontend:** React · TypeScript · Vite · Leaflet · framer-motion
**Data:** DIOR-RSVG, VRSBench, DOTA, NWPU-Captions, SECOND-CC, OpenEarthMap, RSVQA-LR, CDVQA,
BigEarthNet — see `data/scripts/` for the download scripts and `CLAUDE.md` for per-dataset notes.

## Project structure

```
backend/app/            FastAPI app + orchestrator
  orchestrator/          validation, LLM routing, execution, answer composer, trace
  specialists/           adapters wiring models/ into the tool registry (one per tool)
  gee/  gis/             Earth Engine layers and scoring; Esri map-area capture
backend/tests/          pytest suite — orchestrator, API, composer and every specialist
frontend/src/           React + Vite + TypeScript single-page app
models/                 standalone specialist wrappers (grounding, captioning, landcover, ...)
notebooks/              Kaggle fine-tuning notebooks
data/scripts/           dataset download scripts
```

## Status and what's next

All five mandatory capabilities have a working end-to-end implementation through the UI (change
description and optical–SAR analysis only at a basic level); see [`CLAUDE.md`](CLAUDE.md) for the detailed
current-state writeup. The largest gaps against the spec are
an end-to-end benchmark harness (VRSBench, RSVQA, CDVQA), a change-*description* model and CDVQA
evaluation, a learned optical–SAR fusion stage and sensor-robust modality handling (e.g. a single-band
panchromatic image is currently guessed to be SAR), any use of BigEarthNet, and our own VQA fine-tune.
