# GCP LoRA Character Dataset Generator

Revamp of [lovisdotio/NanoBananaLoraDatasetGenerator](https://github.com/lovisdotio/NanoBananaLoraDatasetGenerator).

Auth/generation uses the **same path as onestopvideo**:

```text
Browser UI  →  local python server.py  →  google.genai Client(vertexai=True)
                                         →  Application Default Credentials
```

No Google API keys or pasted OAuth tokens in the browser.

## Run (LAN, port 11904)

```bash
cd D:\Projects\NanoBananaLoraDatasetGenerator-GCP

# Once per machine (same as onestopvideo):
gcloud auth application-default login
gcloud config set project llmapi-503100

python server.py
# binds 0.0.0.0:11904
```

Open:

- Local: http://127.0.0.1:11904/
- LAN: http://&lt;host-lan-ip&gt;:11904/

`server.py` auto-loads `D:\Projects\onestopvideo\.env` for `OSV_PROJECT_ID` / `OSV_VERTEX_*`.

## Models

| Role | Model IDs |
|------|-----------|
| Image | `gemini-3.1-flash-image` (Nano Banana 2), `gemini-3.1-flash-lite-image` (2 Lite), `gemini-3-pro-image` (Pro) |
| LLM / captions | `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite` |

## Modes

1. **Character LoRA** — multi-slot refs + curated shot pack
2. Pair / Single / Reference / Import+Edit

## Offline check

```bash
node selfcheck.mjs
```
