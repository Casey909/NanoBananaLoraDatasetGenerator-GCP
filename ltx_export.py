"""Export character dataset into Lightricks LTX-2.3 image-only train pack."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAIN_SIZE = 768
BUCKET = f"{TRAIN_SIZE}x{TRAIN_SIZE}x1"

LOW_VRAM_YAML = """# LTX-2.3 character LoRA — image-only low VRAM attempt (16–32GB)
# Based on official t2v_lora_low_vram defaults. Bucket: 768x768x1

lora:
  rank: 16
  alpha: 16
  dropout: 0.0

optimization:
  learning_rate: 1.0e-4
  steps: 1500
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: adamw8bit
  scheduler_type: linear
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_dtype: bfloat16
  quantization: int8
  load_text_encoder_in_8bit: true

# Fill these locally / via env before training:
# model_path, text_encoder_path, output_dir
"""

README_TRAIN = """# LTX 2.3 Character LoRA Train Pack

This folder is an **images-only** Lightricks LTX-2.3 dataset.

## Requirements

- Width/height divisible by 32 → images are **768×768**
- Frame count `8n+1` → bucket **768x768x1**
- Homogeneous images only (do not mix videos)
- Unique trigger in `trigger.txt`

## Env / paths

Set before training:

```text
LTX_TRAINER_ROOT=path/to/LTX-2   # contains packages/ltx-trainer
LTX_MODEL_PATH=path/to/ltx-2.3-dev.safetensors
GEMMA_PATH=path/to/gemma
```

## Commands

```bash
# from LTX trainer package
uv run python scripts/process_dataset.py dataset.json ^
  --resolution-buckets "768x768x1" ^
  --model-path %LTX_MODEL_PATH% ^
  --text-encoder-path %GEMMA_PATH% ^
  --lora-trigger "%TRIGGER%"

# then train with train_config.yaml (low VRAM)
```

Or use the app **Train Locally (LTX 2.3)** button.

## Note on 16GB VRAM

Official low-VRAM profile targets ~32GB. Local 16GB may OOM; this pack is still valid for a 32GB+/cloud run.
"""


def default_trigger(slug: str, name: str = "") -> str:
    raw = (slug or name or "char").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_") or "char"
    return f"ohwx_{raw}"


def _square_resize(img, size: int = TRAIN_SIZE):
    from PIL import Image

    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def export_ltx_pack(
    character_dir: Path,
    *,
    trigger: str | None = None,
    train_size: int = TRAIN_SIZE,
    min_images: int = 1,
) -> dict[str, Any]:
    """Build ltx_train/ under character_dir from dataset/*.png + *.json/*.txt."""
    from PIL import Image

    character_dir = Path(character_dir)
    dataset = character_dir / "dataset"
    meta_path = character_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    slug = meta.get("slug") or character_dir.name
    name = meta.get("name") or slug
    trigger = (trigger or default_trigger(slug, name)).strip()

    items: list[dict[str, Any]] = []
    for jp in sorted(dataset.glob("*.json")):
        try:
            item = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if item.get("status") and item.get("status") != "ok":
            continue
        img_path = dataset / f"{jp.stem}.png"
        if not img_path.exists():
            # try jpg
            img_path = dataset / f"{jp.stem}.jpg"
        if not img_path.exists():
            continue
        txt_path = dataset / f"{jp.stem}.txt"
        caption = item.get("ltxCaption") or item.get("text") or ""
        if txt_path.exists() and not caption:
            caption = txt_path.read_text(encoding="utf-8").strip()
        if trigger and trigger not in caption:
            caption = f"{trigger}, {caption}" if caption else trigger
        items.append({"id": jp.stem, "img": img_path, "caption": caption, "tag": item.get("tag") or ""})

    if len(items) < min_images:
        raise ValueError(f"Need at least {min_images} OK images to export, found {len(items)}")

    out = character_dir / "ltx_train"
    images_dir = out / "images"
    captions_dir = out / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # clear previous images
    for old in images_dir.glob("*"):
        old.unlink(missing_ok=True)
    for old in captions_dir.glob("*"):
        old.unlink(missing_ok=True)

    rows: list[dict[str, str]] = []
    for i, it in enumerate(items, start=1):
        stem = f"{i:04d}"
        dest = images_dir / f"{stem}.jpg"
        with Image.open(it["img"]) as im:
            im = im.convert("RGB")
            im = _square_resize(im, train_size)
            im.save(dest, format="JPEG", quality=92, optimize=True)
            if im.size != (train_size, train_size):
                raise RuntimeError(f"Export resize failed for {stem}: {im.size}")
        caption = it["caption"]
        (captions_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
        rows.append({"caption": caption, "media_path": f"images/{stem}.jpg"})

    (out / "dataset.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "trigger.txt").write_text(trigger + "\n", encoding="utf-8")
    (out / "train_config.yaml").write_text(LOW_VRAM_YAML, encoding="utf-8")
    readme = README_TRAIN.replace("%TRIGGER%", trigger)
    (out / "README_TRAIN.md").write_text(readme, encoding="utf-8")
    manifest = {
        "slug": slug,
        "name": name,
        "trigger": trigger,
        "bucket": f"{train_size}x{train_size}x1",
        "imageCount": len(rows),
        "trainSize": train_size,
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "path": str(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
