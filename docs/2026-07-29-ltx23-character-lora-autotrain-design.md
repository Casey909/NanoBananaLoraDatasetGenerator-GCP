# LTX 2.3 Character LoRA Auto-Train Design

**Date:** 2026-07-29  
**Status:** Approved for planning (pending user review of this written spec)  
**Primary surface:** `NanoBananaLoraDatasetGenerator-GCP` / `gcp-lora-character-dataset`  
**Downstream consumer:** LTX 2.3 (ComfyUI / onestopvideo video worker)

## 1. Goal

End-to-end flow:

1. Generate **90 diverse** character stills from a few reference photos.
2. Export them in **Lightricks LTX-2.3 trainer format** at the **required train resolution**.
3. **Train locally first** (user GPU: ~16GB dedicated + Windows shared GPU memory).
4. Produce a `.safetensors` LoRA that loads on **LTX 2.3** for identity-consistent video.

Non-goals for this iteration:

- IC-LoRA (depth/pose/canny control adapters) — different product.
- Mixing images + videos in one train run.
- Guaranteeing successful train on 16GB dedicated VRAM (attempt local-first; document OOM fallback).

## 2. Hardware reality (local-first)

| Capability | 16GB dedicated | Notes |
|------------|----------------|-------|
| LTX 2.3 **inference** (FP8) | Supported | Matches onestopvideo / ComfyUI use |
| LTX 2.3 **LoRA training** | Unofficial / risky | Official low-VRAM config targets **~32GB**; field reports ~31GB even with INT8 + checkpointing |
| Windows “96GB total GPU memory” | Mostly **shared system RAM** | Helps paging; not equal to dedicated CUDA VRAM |

**Policy:** Ship local train button + low-VRAM config. On CUDA OOM, keep the export pack unchanged and show a clear fallback (32GB+ local or short cloud rental) using the same files/config.

## 3. Architecture

```text
Refs (cropped) → backend job (90 stills)
              → captions + trigger
              → ltx_train/ export (768×768 + dataset.json + train_config.yaml)
              → local ltx-trainer preprocess + train
              → loras/<slug>_ltx23.safetensors
              → LTX 2.3 LoRA loader (ComfyUI / onestopvideo)
```

Components:

| Unit | Responsibility |
|------|----------------|
| Shot recipe (90) | Deterministic diversity templates including shirtless/swimwear |
| Image job runner | Existing backend jobs; default count 90; 1:1 aspect preferred |
| LTX export | Resize to 768×768; write `dataset.json`, captions, config, README |
| Local trainer wrapper | Invoke Lightricks `ltx-trainer` preprocess + train; stream logs |
| LoRA registry path | Write trained weights under character folder + optional Comfy `models/loras` copy hint |

## 4. Ninety-image recipe

Total **90** unique stills (no near-duplicate spam):

| Block | Count | Content |
|------|------:|---------|
| Face identity | 22 | front, ¾ L/R, profile L/R, up/down, expressions |
| Head & shoulders | 14 | lighting / micro wardrobe variants |
| Upper body (clothed) | 14 | poses, hands, casual outfits |
| Full body (clothed) | 14 | front/side/back, walk/stand/sit |
| Shirtless / swimwear (body lock) | 14 | shirtless torso; swim briefs/trunks or one-piece/bikini as gender-appropriate; beach/pool/studio; tasteful, non-explicit |
| Context / hard cases | 12 | indoor/outdoor, dramatic light, mild occlusion, same identity |

Generation defaults:

- Mode: character LoRA
- Count: **90**
- Aspect: **1:1** (clean square for train resize)
- Image size: user choice (1K/2K/4K per model limits); quality can exceed train size
- Captions: `trigger, <shot description>, <appearance / wardrobe cues>`

## 5. LTX 2.3 train format

### 5.1 Folder layout

```text
data/characters/<slug>/ltx_train/
  images/0001.jpg … 0090.jpg      # exactly 768×768 JPEG
  dataset.json                    # Lightricks schema
  captions/0001.txt …             # optional mirror of captions
  trigger.txt
  train_config.yaml               # low-VRAM LTX 2.3 LoRA
  README_TRAIN.md
loras/<slug>_ltx23.safetensors    # after successful train (or under ltx_train/output/)
```

### 5.2 `dataset.json` schema

```json
[
  {
    "caption": "ohwx_leo, close-up front face portrait, soft studio light, plain background",
    "media_path": "images/0001.jpg"
  }
]
```

Rules (official Lightricks guidance):

- Images only (homogeneous) — JPEG or PNG
- Resolution bucket: **`768x768x1`** (W/H divisible by 32; frames = `8n+1`, so `1` for stills)
- Unique `--lora-trigger` during preprocess
- Do not mix video clips into this dataset

### 5.3 Export resolution pipeline

1. Read generated dataset images from `dataset/*.png` (or job outputs).
2. Center-crop / contain to square without distorting identity (prefer cover + center).
3. Resize to **exactly 768×768**.
4. Save JPEG quality ~92 as `ltx_train/images/NNNN.jpg`.
5. Build captions with trigger prefix; write `dataset.json`.

### 5.4 Local train defaults (`train_config.yaml`)

Based on official `t2v_lora_low_vram.yaml` adapted for image-only character:

| Setting | Value |
|---------|-------|
| Bucket | `768x768x1` |
| Rank / alpha | 16 / 16 |
| Optimizer | `adamw8bit` |
| Gradient checkpointing | true |
| Quantization | INT8 / text encoder 8-bit where supported |
| Learning rate | `1e-4` |
| Steps | 1500 (checkpoint every 250; user picks best) |
| Batch size | 1 |
| Precision | bf16 |

UI actions:

- **Export LTX Train Pack**
- **Train Locally (LTX 2.3)** → preprocess (`process_dataset.py`) → train → write LoRA path + logs

## 6. Using the LoRA with LTX 2.3

1. Copy `*.safetensors` into ComfyUI `models/loras/` (or onestopvideo LTX LoRA directory).
2. Load with the **standard character LoRA loader** (not `LTXICLoRALoaderModelOnly` — that path is for IC-LoRA control).
3. Include the trigger token in the prompt.
4. Start strength ~**0.7–1.0**; increase if identity is weak, decrease if motion/style collapses.

Note: LTX 2.0 LoRAs are **not** compatible with 2.3 — always train against the 2.3 base/dev checkpoint used at inference.

## 7. Error handling

| Failure | Behavior |
|---------|----------|
| Gen job partial failures | Keep successes; export only `status=ok` images; warn if under 60 |
| Fewer than 60 OK images | Block train; allow export for manual curation |
| Train CUDA OOM | Fail job with actionable message; pack remains valid for 32GB+/cloud |
| Missing LTX model / Gemma paths | Settings panel: require `LTX_MODEL_PATH`, `GEMMA_PATH`, `LTX_TRAINER_ROOT` |
| Invalid trigger | Auto-slugify unique trigger from character name (`ohwx_<slug>`) |

## 8. Verification

- Selfcheck: 90 templates present; export writes 768×768; `dataset.json` length matches images.
- Manual: generate small smoke (e.g. 4 images) → export → verify dimensions → optional 1-step train smoke if models installed.
- After full train: LoRA file exists; prompt with/without trigger differs in identity lock on LTX 2.3.

## 9. Implementation scope (next plan)

1. Expand shot templates to 90 (incl. shirtless/swimwear) in `jobs.py` / `character.js`.
2. Default UI count 90; raise max; lock aspect hint to 1:1 for LTX export.
3. Add export API + UI for `ltx_train/` pack.
4. Add local train wrapper + log streaming + OOM messaging.
5. README: model download paths, one-command train, ComfyUI load steps.

## 10. Open assumptions (explicit)

- User will install Lightricks `LTX-2` / `ltx-trainer` and download LTX 2.3 **dev** + Gemma locally; the app does not bundle multi-GB weights.
- Shirtless/swimwear shots are **tasteful body-proportion references**, not explicit adult content.
- If local 16GB train fails, the deliverable pack is still the primary success for cloud/32GB train.
