"""Local LTX-2.3 LoRA train wrapper (preprocess + train subprocess)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def _env_paths() -> dict[str, str]:
    return {
        "trainerRoot": (os.environ.get("LTX_TRAINER_ROOT") or "").strip(),
        "modelPath": (os.environ.get("LTX_MODEL_PATH") or "").strip(),
        "gemmaPath": (os.environ.get("GEMMA_PATH") or os.environ.get("LTX_TEXT_ENCODER_PATH") or "").strip(),
    }


def train_status(character_dir: Path) -> dict[str, Any]:
    status_path = Path(character_dir) / "ltx_train" / "train_status.json"
    if not status_path.exists():
        paths = _env_paths()
        ready = bool(paths["trainerRoot"] and paths["modelPath"] and paths["gemmaPath"])
        return {
            "status": "idle",
            "pathsConfigured": ready,
            "paths": {k: bool(v) for k, v in paths.items()},
            "hint": None
            if ready
            else "Set LTX_TRAINER_ROOT, LTX_MODEL_PATH, GEMMA_PATH then retry Train Locally.",
        }
    return json.loads(status_path.read_text(encoding="utf-8"))


def _write_status(character_dir: Path, payload: dict[str, Any]) -> None:
    out = Path(character_dir) / "ltx_train"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train_status.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_LOCK = threading.Lock()
_RUNNING: dict[str, subprocess.Popen] = {}


def start_local_train(character_dir: Path, log_fn=None) -> dict[str, Any]:
    character_dir = Path(character_dir)
    slug = character_dir.name
    pack = character_dir / "ltx_train"
    dataset = pack / "dataset.json"
    if not dataset.exists():
        raise FileNotFoundError("Export LTX train pack first (missing ltx_train/dataset.json)")

    paths = _env_paths()
    missing = [k for k, v in paths.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing env for local train: "
            + ", ".join(missing)
            + ". Set LTX_TRAINER_ROOT, LTX_MODEL_PATH, GEMMA_PATH."
        )

    trainer_root = Path(paths["trainerRoot"])
    process_script = trainer_root / "packages" / "ltx-trainer" / "scripts" / "process_dataset.py"
    if not process_script.exists():
        # alternate layout
        process_script = trainer_root / "scripts" / "process_dataset.py"
    if not process_script.exists():
        raise FileNotFoundError(f"process_dataset.py not found under {trainer_root}")

    trigger = (pack / "trigger.txt").read_text(encoding="utf-8").strip() or f"ohwx_{slug}"
    log_path = pack / "train.log"
    status = {
        "status": "running",
        "startedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "slug": slug,
        "logFile": str(log_path),
        "message": "Starting preprocess + train (local low-VRAM). 16GB may OOM.",
    }
    _write_status(character_dir, status)

    def _runner():
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"# LTX local train {slug}\n")
                log.flush()
                cmd_pre = [
                    "python",
                    str(process_script),
                    str(dataset),
                    "--resolution-buckets",
                    "768x768x1",
                    "--model-path",
                    paths["modelPath"],
                    "--text-encoder-path",
                    paths["gemmaPath"],
                    "--lora-trigger",
                    trigger,
                ]
                log.write("PREPROCESS: " + " ".join(cmd_pre) + "\n")
                log.flush()
                if log_fn:
                    log_fn("INFO", "ltx.preprocess_start", slug=slug)
                p = subprocess.Popen(
                    cmd_pre,
                    cwd=str(process_script.parent.parent if "packages" in str(process_script) else trainer_root),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with _LOCK:
                    _RUNNING[slug] = p
                code = p.wait()
                if code != 0:
                    raise RuntimeError(f"preprocess failed exit={code} (see train.log)")

                # Prefer official train entry if present
                train_py = process_script.parent / "train.py"
                if not train_py.exists():
                    train_py = trainer_root / "packages" / "ltx-trainer" / "scripts" / "train.py"
                config = pack / "train_config.yaml"
                if train_py.exists():
                    cmd_train = ["python", str(train_py), str(config)]
                    log.write("\nTRAIN: " + " ".join(cmd_train) + "\n")
                    log.flush()
                    if log_fn:
                        log_fn("INFO", "ltx.train_start", slug=slug)
                    p2 = subprocess.Popen(
                        cmd_train,
                        cwd=str(train_py.parent.parent if "packages" in str(train_py) else trainer_root),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    with _LOCK:
                        _RUNNING[slug] = p2
                    code2 = p2.wait()
                    if code2 != 0:
                        raise RuntimeError(f"train failed exit={code2} (see train.log) — 16GB OOM is common")
                else:
                    log.write(
                        "\nNo train.py found. Preprocess done. "
                        "Run ltx-trainer manually with train_config.yaml.\n"
                    )

            # collect safetensors if any
            outs = list(pack.glob("**/*.safetensors")) + list((character_dir / "loras").glob("*.safetensors"))
            status_ok = {
                "status": "completed",
                "finishedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "slug": slug,
                "logFile": str(log_path),
                "loras": [str(p) for p in outs],
                "message": "Train finished" if outs else "Finished (check train.log / output for .safetensors)",
            }
            _write_status(character_dir, status_ok)
            if log_fn:
                log_fn("INFO", "ltx.train_done", slug=slug, loras=len(outs))
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            oom = "out of memory" in msg.lower() or "cuda" in msg.lower() and "memory" in msg.lower()
            status_err = {
                "status": "failed",
                "finishedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "slug": slug,
                "logFile": str(log_path),
                "error": msg[:800],
                "oomLikely": oom,
                "message": (
                    "CUDA OOM on 16GB is expected for LTX-2.3. Pack is intact — retry on 32GB+ GPU."
                    if oom
                    else msg[:400]
                ),
            }
            _write_status(character_dir, status_err)
            if log_fn:
                log_fn("ERROR", "ltx.train_failed", slug=slug, error=msg[:300])
        finally:
            with _LOCK:
                _RUNNING.pop(slug, None)

    threading.Thread(target=_runner, daemon=True, name=f"ltx-train-{slug}").start()
    time.sleep(0.05)
    return train_status(character_dir)
