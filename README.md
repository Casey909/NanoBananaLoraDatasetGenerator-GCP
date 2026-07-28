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
```

## Features

- Backend jobs with auto-resume on 429
- Live log/status polling (survives page hide)
- Per-image refine textarea + Regenerate
- beforeunload confirm while a job is active
- ZIP download from saved folder files

## Models

| Role | IDs |
|------|-----|
| Image | `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, `gemini-3-pro-image` |
| LLM | `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite` |
