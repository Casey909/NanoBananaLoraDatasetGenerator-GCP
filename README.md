# GCP LoRA Character Dataset Generator

Backend-first LoRA pack generator using the **same Vertex ADC path as onestopvideo**.

```text
Browser  →  python server.py (background jobs)
         →  google.genai Client(vertexai=True) + ADC
         →  data/characters/<slug>/{refs,dataset,jobs}/
```

Hiding or closing the browser tab does **not** stop generation (server process must keep running). The UI polls job status/logs and reloads results from disk.

## Run

```bash
cd D:\Projects\NanoBananaLoraDatasetGenerator-GCP
gcloud auth application-default login
python server.py
# http://0.0.0.0:11904  (LAN OK)
```

Uses `D:\Projects\onestopvideo\.env` for `OSV_PROJECT_ID` / `OSV_VERTEX_*` when present.

## Character folders

```
data/characters/<slug>/
  meta.json
  refs/face_front.jpg
  dataset/0001.png
  dataset/0001.txt
  dataset/0001.json
  jobs/<jobId>.json
  ltx_train/
    images/0001.jpg …   # exactly 768×768
    dataset.json        # Lightricks LTX-2.3 schema
    trigger.txt
    train_config.yaml
    README_TRAIN.md
```

## LTX 2.3 train pack

1. Generate **90** character shots (default) with trigger word set (e.g. `ohwx_leo`).
2. Click **Export LTX Train Pack** → writes `ltx_train/` at **768×768** + `dataset.json`.
3. Optional **Train Locally (LTX 2.3)** after setting:

```text
set LTX_TRAINER_ROOT=D:\path\to\LTX-2
set LTX_MODEL_PATH=D:\path\to\ltx-2.3-dev.safetensors
set GEMMA_PATH=D:\path\to\gemma
```

Official low-VRAM training targets ~32GB; 16GB may OOM — the export pack remains valid for a larger GPU.

Load the resulting `.safetensors` in ComfyUI / onestopvideo LTX 2.3 with the **standard LoRA loader** (not IC-LoRA), and include the trigger in prompts.

## Features

- 90-shot diversity recipe (faces, body, shirtless/swimwear, context)
- Backend jobs with auto-resume on 429
- Live log/status polling (survives page hide)
- Per-image refine textarea + Regenerate
- beforeunload confirm while a job is active
- Crop tool on reference upload (aspect presets, drag/resize)
- LTX 2.3 export + local train wrapper
- ZIP download from saved folder files

## Models

| Role | IDs |
|------|-----|
| Image | `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, `gemini-3-pro-image` |
| LLM | `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite` |
