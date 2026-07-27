# GCP LoRA Character Dataset Generator

Revamp of [lovisdotio/NanoBananaLoraDatasetGenerator](https://github.com/lovisdotio/NanoBananaLoraDatasetGenerator).

FAL.ai is replaced with **Google Gemini / Vertex AI** so you can generate LoRA training packs from a few character reference images (face, body, etc.).

## Models

### Image (Nano Banana)

| UI label | Model ID |
|----------|----------|
| Nano Banana 2 | `gemini-3.1-flash-image` |
| Nano Banana 2 Lite | `gemini-3.1-flash-lite-image` |
| Nano Banana Pro | `gemini-3-pro-image` |

### LLM / captions

| UI label | Model ID |
|----------|----------|
| Gemini 3.6 Flash | `gemini-3.6-flash` |
| Gemini 3.5 Flash | `gemini-3.5-flash` |
| Gemini 3.5 Flash Lite | `gemini-3.5-flash-lite` |

Flash models are used for prompt expansion and optional vision captions. Image generation uses the Nano Banana family only.

## Modes

1. **Character LoRA** (default) — multi-slot refs (front face required) + curated pose/angle pack
2. **Pair** — START → END edit pairs
3. **Single** — style/aesthetic images
4. **Reference** — variations from one image
5. **Import + Edit** — batch-edit a local folder

## Quick start

```bash
cd gcp-lora-character-dataset
python -m http.server 8765
# open http://localhost:8765
```

Or:

```bash
npx --yes serve .
```

1. Click 🔑 and add a [Google AI Studio API key](https://aistudio.google.com/apikey)  
   **or** Vertex AI project ID + `gcloud auth print-access-token`
2. Upload face (+ optional body / side / outfit refs)
3. Choose image + LLM models
4. Start generation → Download ZIP

## Output

```
gcp_lora_dataset_TIMESTAMP.zip
├── 0001.png
├── 0001.txt
├── 0002.png
├── 0002.txt
└── ...
```

Pair / import-edit modes write `NNNN_start.png` + `NNNN_end.png` + `NNNN.txt`.

## Files

```
gcp-lora-character-dataset/
├── index.html      # UI
├── app.js          # generation flows
├── gemini.js       # Google AI + Vertex client
├── character.js    # LoRA shot presets + ref slots
├── style.css
├── selfcheck.mjs   # offline sanity checks
└── README.md
```

## Notes

- Credentials stay in `localStorage` and are sent only to Google endpoints.
- Nano Banana 2 Lite is 1K-only and weaker with many refs — prefer **Nano Banana 2** or **Pro** for character consistency.
- Start with parallel = 1–2 to avoid rate limits.
- Vertex tokens expire; refresh with `gcloud auth print-access-token`.

## License

MIT (same spirit as the upstream project).
