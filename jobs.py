"""Background job runner + character folder storage for LoRA dataset generation."""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "characters"

REF_SLOTS = [
    "face_front",
    "face_side",
    "body_full",
    "body_upper",
    "extra",
]

CHARACTER_SHOT_TEMPLATES = [
    {"tag": "face_front", "prompt": "close-up front face portrait, eyes looking at camera, neutral expression, soft even studio lighting, plain light gray background, photorealistic, high detail skin and facial features"},
    {"tag": "face_three_quarter_left", "prompt": "three-quarter view face portrait angled left, soft smile, natural daylight, shallow depth of field, plain background"},
    {"tag": "face_three_quarter_right", "prompt": "three-quarter view face portrait angled right, calm expression, soft Rembrandt lighting, plain background"},
    {"tag": "face_profile_left", "prompt": "strict left profile headshot, neutral expression, clean silhouette, soft studio lighting, plain background"},
    {"tag": "face_profile_right", "prompt": "strict right profile headshot, neutral expression, clean silhouette, soft studio lighting, plain background"},
    {"tag": "face_looking_up", "prompt": "headshot looking slightly upward, soft hopeful expression, overhead soft light, plain background"},
    {"tag": "face_looking_down", "prompt": "headshot looking slightly downward, contemplative expression, soft side light, plain background"},
    {"tag": "expression_smile", "prompt": "close-up portrait with a natural genuine smile, eyes engaged, soft beauty lighting, plain background"},
    {"tag": "expression_serious", "prompt": "close-up portrait with a serious focused expression, cinematic soft key light, plain background"},
    {"tag": "expression_laugh", "prompt": "close-up portrait mid-laugh, joyful expression, natural outdoor light, soft bokeh background"},
    {"tag": "upper_body_front", "prompt": "waist-up portrait facing camera, relaxed arms, casual clothing, soft studio lighting, plain background"},
    {"tag": "upper_body_side", "prompt": "waist-up three-quarter pose, one hand visible, casual clothing, soft window light, plain background"},
    {"tag": "full_body_front", "prompt": "full body standing front view head to toe, natural stance, full outfit visible, even lighting, plain seamless background"},
    {"tag": "full_body_side", "prompt": "full body standing side view, natural posture, full outfit visible, even lighting, plain seamless background"},
    {"tag": "full_body_back", "prompt": "full body standing back view looking over shoulder toward camera, full outfit visible, even lighting, plain background"},
    {"tag": "sitting_pose", "prompt": "character sitting casually on a simple stool, waist-up framing, relaxed pose, soft studio lighting, plain background"},
    {"tag": "walking_pose", "prompt": "full body walking pose mid-stride, natural motion, outdoor soft daylight, simple blurred background"},
    {"tag": "hands_near_face", "prompt": "close portrait with one hand gently near the face/chin, elegant pose, soft beauty lighting, plain background"},
    {"tag": "different_outfit", "prompt": "waist-up portrait in a different casual outfit than the references, same person identity preserved, soft studio lighting, plain background"},
    {"tag": "outdoor_context", "prompt": "outdoor environmental portrait, upper body, natural daylight, park or street soft bokeh, identity consistent with references"},
    {"tag": "indoor_context", "prompt": "indoor lifestyle portrait, upper body, warm interior lighting, simple room background, identity consistent with references"},
    {"tag": "dramatic_light", "prompt": "dramatic cinematic portrait, strong contrast lighting, close face framing, dark gradient background, identity preserved"},
    {"tag": "soft_beauty", "prompt": "beauty headshot, soft diffused light, clean skin detail, gentle catchlights in eyes, light gray seamless background"},
    {"tag": "wide_angle_full", "prompt": "full body wide shot standing centered, head to toe visible, even lighting, plain seamless studio background"},
]

LogFn = Callable[..., None]


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "").strip().lower()).strip("-")
    return s or f"char-{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JobStore:
    def __init__(self, data_root: Path = DATA_ROOT, log_fn: LogFn | None = None):
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.log_fn = log_fn or (lambda *a, **k: None)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}

    # ---- filesystem helpers ----

    def character_dir(self, slug: str) -> Path:
        return self.data_root / slug

    def ensure_character(self, name: str, slug: str | None = None) -> dict[str, Any]:
        slug = slugify(slug or name)
        cdir = self.character_dir(slug)
        (cdir / "refs").mkdir(parents=True, exist_ok=True)
        (cdir / "dataset").mkdir(parents=True, exist_ok=True)
        (cdir / "jobs").mkdir(parents=True, exist_ok=True)
        meta_path = cdir / "meta.json"
        meta = {
            "name": name.strip() or slug,
            "slug": slug,
            "createdAt": now_iso(),
            "updatedAt": now_iso(),
        }
        if meta_path.exists():
            try:
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["createdAt"] = old.get("createdAt") or meta["createdAt"]
                meta["name"] = name.strip() or old.get("name") or slug
            except Exception:
                pass
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def list_characters(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.data_root.glob("*/meta.json")):
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                slug = meta.get("slug") or p.parent.name
                refs = sorted([x.stem for x in (p.parent / "refs").glob("*.*")])
                dataset = sorted((p.parent / "dataset").glob("*.png"))
                out.append(
                    {
                        **meta,
                        "slug": slug,
                        "refs": refs,
                        "imageCount": len(dataset),
                    }
                )
            except Exception:
                continue
        return out

    def get_character(self, slug: str) -> dict[str, Any]:
        cdir = self.character_dir(slug)
        meta_path = cdir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Character not found: {slug}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        refs = {}
        for slot in REF_SLOTS:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                f = cdir / "refs" / f"{slot}{ext}"
                if f.exists():
                    refs[slot] = f"/api/files/{slug}/refs/{f.name}"
                    break
        items = []
        for meta_item in sorted((cdir / "dataset").glob("*.json")):
            try:
                item = json.loads(meta_item.read_text(encoding="utf-8"))
                items.append(item)
            except Exception:
                continue
        return {**meta, "slug": slug, "refs": refs, "items": items}

    def save_ref_data_url(self, slug: str, slot: str, data_url: str) -> str:
        if slot not in REF_SLOTS:
            raise ValueError(f"Invalid ref slot: {slot}")
        # Preserve existing display name; empty name keeps meta.json name.
        self.ensure_character("", slug)
        # Lazy import from server helpers via callback injection later; decode here.
        from server import _decode_data_url, _prepare_ref_image  # circular-safe at runtime

        mime, blob = _decode_data_url(data_url)
        mime, blob = _prepare_ref_image(blob, mime)
        ext = ".jpg" if mime == "image/jpeg" else ".png" if mime == "image/png" else ".jpg"
        # clear old extensions
        for old in (self.character_dir(slug) / "refs").glob(f"{slot}.*"):
            old.unlink(missing_ok=True)
        path = self.character_dir(slug) / "refs" / f"{slot}{ext}"
        path.write_bytes(blob)
        # touch updatedAt without renaming
        meta_path = self.character_dir(slug) / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["updatedAt"] = now_iso()
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception:
                pass
        self.log_fn("INFO", "character.ref_saved", slug=slug, slot=slot, bytes=len(blob), path=str(path))
        return f"/api/files/{slug}/refs/{path.name}"

    def load_ref_data_urls(self, slug: str, max_refs: int = 4) -> list[str]:
        urls: list[str] = []
        refs_dir = self.character_dir(slug) / "refs"
        for slot in REF_SLOTS:
            if len(urls) >= max_refs:
                break
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                f = refs_dir / f"{slot}{ext}"
                if f.exists():
                    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png" if ext == ".png" else "image/webp"
                    b64 = base64.b64encode(f.read_bytes()).decode("ascii")
                    urls.append(f"data:{mime};base64,{b64}")
                    break
        return urls

    def resolve_file(self, slug: str, rel: str) -> Path:
        # rel like refs/face_front.jpg or dataset/0001.png
        path = (self.character_dir(slug) / rel).resolve()
        root = self.character_dir(slug).resolve()
        if not str(path).startswith(str(root)):
            raise PermissionError("Invalid path")
        if not path.exists():
            raise FileNotFoundError(rel)
        return path

    def _job_path(self, slug: str, job_id: str) -> Path:
        return self.character_dir(slug) / "jobs" / f"{job_id}.json"

    def _persist_job(self, job: dict[str, Any]) -> None:
        slug = job["characterSlug"]
        path = self._job_path(slug, job["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _append_log(self, job: dict[str, Any], level: str, message: str, **extra: Any) -> None:
        entry = {"ts": now_iso(), "level": level, "message": message, **extra}
        logs = job.setdefault("logs", [])
        logs.append(entry)
        if len(logs) > 2000:
            del logs[: len(logs) - 2000]
        job["updatedAt"] = now_iso()
        self._persist_job(job)
        self.log_fn(level, "job.log", jobId=job["id"], message=message, **extra)

    # ---- job lifecycle ----

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return json.loads(json.dumps(job))  # copy
        # disk fallback
        for meta in self.data_root.glob("*/jobs/*.json"):
            if meta.stem == job_id:
                try:
                    job = json.loads(meta.read_text(encoding="utf-8"))
                    with self._lock:
                        self._jobs[job_id] = job
                    return json.loads(json.dumps(job))
                except Exception:
                    return None
        return None

    def list_jobs(self, slug: str | None = None) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []

        def _row(job: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": job["id"],
                "characterSlug": job["characterSlug"],
                "status": job["status"],
                "completed": job.get("completed", 0),
                "failed": job.get("failed", 0),
                "total": job.get("total", 0),
                "updatedAt": job.get("updatedAt"),
            }

        with self._lock:
            for job in self._jobs.values():
                if slug and job.get("characterSlug") != slug:
                    continue
                seen.add(job["id"])
                out.append(_row(job))

        for path in self.data_root.glob(f"{slug or '*'}/jobs/*.json"):
            jid = path.stem
            if jid in seen:
                continue
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if slug and job.get("characterSlug") != slug:
                    continue
                out.append(_row(job))
            except Exception:
                continue
        return sorted(out, key=lambda x: x.get("updatedAt") or "", reverse=True)

    def latest_job_id(self, slug: str) -> str | None:
        jobs = self.list_jobs(slug)
        return jobs[0]["id"] if jobs else None

    def stop_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            flag = self._stop_flags.get(job_id)
            job = self._jobs.get(job_id)
            if flag:
                flag.set()
            if job and job.get("status") == "running":
                job["status"] = "stopping"
                self._append_log(job, "WARN", "Stop requested")
                self._persist_job(job)
            return self.get_job(job_id) or {"id": job_id, "status": "unknown"}

    def create_and_start_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = (payload.get("characterName") or payload.get("name") or "").strip()
        slug = slugify(payload.get("characterSlug") or name)
        if not name and not payload.get("characterSlug"):
            raise ValueError("characterName is required")
        meta = self.ensure_character(name or slug, slug)

        refs = self.load_ref_data_urls(slug, max_refs=4)
        mode = payload.get("mode") or "character"
        if mode == "character" and not refs:
            raise ValueError("Upload at least one character reference (face_front) before starting")

        job_id = uuid.uuid4().hex[:12]
        count = max(1, min(40, int(payload.get("count") or payload.get("numPairs") or 20)))
        settings = {
            "mode": mode,
            "imageModel": payload.get("imageModel") or "gemini-3.1-flash-image",
            "llmModel": payload.get("llmModel") or "gemini-3.6-flash",
            "aspectRatio": payload.get("aspectRatio") or "1:1",
            "imageSize": payload.get("imageSize") or "1K",
            "triggerWord": (payload.get("triggerWord") or "").strip(),
            "theme": (payload.get("theme") or "").strip(),
            "useCharacterPresets": payload.get("useCharacterPresets", True),
            "useVisionCaption": bool(payload.get("useVisionCaption")),
            "autoResume": payload.get("autoResume", True),
            "maxConcurrent": max(1, min(3, int(payload.get("maxConcurrent") or 1))),
            "transformation": (payload.get("transformation") or "").strip(),
            "actionName": (payload.get("actionName") or "").strip(),
        }

        job = {
            "id": job_id,
            "characterName": meta["name"],
            "characterSlug": slug,
            "status": "queued",
            "createdAt": now_iso(),
            "updatedAt": now_iso(),
            "settings": settings,
            "total": count,
            "completed": 0,
            "failed": 0,
            "current": 0,
            "logs": [],
            "items": [],
        }
        with self._lock:
            self._jobs[job_id] = job
            self._stop_flags[job_id] = threading.Event()
            self._persist_job(job)
            t = threading.Thread(target=self._run_job, args=(job_id,), daemon=True, name=f"job-{job_id}")
            self._threads[job_id] = t
            t.start()
        self.log_fn("INFO", "job.started", jobId=job_id, slug=slug, count=count, mode=mode)
        return self.get_job(job_id)

    def regenerate_item(self, job_id: str, item_id: str, prompt: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            raise FileNotFoundError("Job not found")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is required")

        # Run regenerate in a short background thread so UI can poll.
        def _worker():
            with self._lock:
                live = self._jobs.get(job_id)
                if not live:
                    return
                item = next((x for x in live.get("items", []) if x.get("id") == item_id), None)
                if not item:
                    self._append_log(live, "ERROR", f"Regenerate failed: item {item_id} not found")
                    return
                item["status"] = "regenerating"
                item["prompt"] = prompt
                self._append_log(live, "INFO", f"Regenerating #{item_id} with refined prompt")
                self._persist_job(live)
            try:
                self._generate_one_item(job_id, item_id, prompt_override=prompt, replace=True)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    live = self._jobs.get(job_id)
                    if live:
                        item = next((x for x in live.get("items", []) if x.get("id") == item_id), None)
                        if item:
                            item["status"] = "failed"
                            item["error"] = str(exc)
                        self._append_log(live, "ERROR", f"#{item_id} regenerate failed: {exc}")
                        self._persist_job(live)

        threading.Thread(target=_worker, daemon=True, name=f"regen-{job_id}-{item_id}").start()
        return self.get_job(job_id)

    # ---- generation internals ----

    def _stopped(self, job_id: str) -> bool:
        flag = self._stop_flags.get(job_id)
        return bool(flag and flag.is_set())

    def _build_prompts(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        from server import generate_text_vertex, parse_json_array_safe

        settings = job["settings"]
        count = int(job["total"])
        mode = settings["mode"]
        name = job["characterName"]
        theme = settings.get("theme") or ""
        trigger = settings.get("triggerWord") or ""

        if mode == "character" and settings.get("useCharacterPresets", True):
            shots = CHARACTER_SHOT_TEMPLATES[:count]
            prompts = []
            for s in shots:
                theme_bit = f" Theme/context hint: {theme}." if theme else ""
                prompts.append(
                    {
                        "prompt": (
                            f"Using the provided reference images, generate a new photo of {name}: "
                            f"{s['prompt']}.{theme_bit} Maintain identity consistency with all references."
                        ),
                        "tag": s["tag"],
                    }
                )
            return prompts

        # LLM prompt expansion
        if mode == "character":
            user = (
                f'Generate {count} unique LoRA training shot prompts for character "{name}".\n'
                f'Theme/notes: "{theme}"\n'
                f'Keep exact identity. Trigger word hint: "{trigger}".\n'
                'Return ONLY JSON array: [{"prompt":"...","tag":"short_tag"}]'
            )
        elif mode == "pair":
            user = (
                f'Generate {count} unique prompt pairs for theme: "{theme}"\n'
                f'Transformation: "{settings.get("transformation")}"\n'
                'Return ONLY JSON array: [{"base_prompt":"...","edit_prompt":"...","action_name":"..."}]'
            )
        else:
            user = (
                f'Generate {count} unique image prompts for theme: "{theme}"\n'
                'Return ONLY JSON array: [{"prompt":"..."}]'
            )

        text = generate_text_vertex(
            {
                "modelId": settings["llmModel"],
                "userText": user,
                "temperature": 0.8,
                "maxOutputTokens": 8192,
            }
        )["text"]
        arr = parse_json_array_safe(text)
        return arr[:count]

    def _generate_one_item(
        self,
        job_id: str,
        item_id: str | None = None,
        prompt_override: str | None = None,
        replace: bool = False,
        prompt_obj: dict[str, Any] | None = None,
        index: int = 0,
    ) -> dict[str, Any]:
        from server import generate_image_vertex, generate_text_vertex

        with self._lock:
            job = self._jobs[job_id]
            settings = dict(job["settings"])
            slug = job["characterSlug"]

        refs = self.load_ref_data_urls(slug, max_refs=4)
        if prompt_obj is None and item_id:
            with self._lock:
                item = next((x for x in self._jobs[job_id]["items"] if x["id"] == item_id), None)
                if not item:
                    raise FileNotFoundError(item_id)
                prompt_obj = {
                    "prompt": prompt_override or item.get("prompt") or "",
                    "tag": item.get("tag") or "",
                    "base_prompt": item.get("basePrompt"),
                    "edit_prompt": item.get("editPrompt"),
                }
                item_id = item["id"]

        assert prompt_obj is not None
        mode = settings["mode"]
        trigger = settings.get("triggerWord") or ""

        # Auto-resume loop for a single item.
        attempt = 0
        last_err: Exception | None = None
        while attempt < 8:
            attempt += 1
            if self._stopped(job_id) and not replace:
                raise RuntimeError("stopped")
            try:
                if mode == "pair":
                    start = generate_image_vertex(
                        {
                            "modelId": settings["imageModel"],
                            "prompt": prompt_obj.get("base_prompt") or prompt_obj.get("prompt") or "",
                            "aspectRatio": settings["aspectRatio"],
                            "imageSize": settings["imageSize"],
                        }
                    )
                    end = generate_image_vertex(
                        {
                            "modelId": settings["imageModel"],
                            "prompt": prompt_override or prompt_obj.get("edit_prompt") or "",
                            "referenceDataUrls": [f"data:image/png;base64,{start['data']}"],
                            "aspectRatio": settings["aspectRatio"],
                            "imageSize": settings["imageSize"],
                        }
                    )
                    caption = prompt_obj.get("action_name") or prompt_obj.get("edit_prompt") or ""
                    if settings.get("useVisionCaption"):
                        try:
                            caption = generate_text_vertex(
                                {
                                    "modelId": settings["llmModel"],
                                    "userText": "Describe the edit from image1 to image2 in one sentence.",
                                    "imageDataUrls": [
                                        f"data:image/png;base64,{start['data']}",
                                        f"data:image/png;base64,{end['data']}",
                                    ],
                                    "temperature": 0.3,
                                }
                            )["text"]
                        except Exception:
                            pass
                    if trigger:
                        caption = f"{trigger} {caption}".strip()
                    return self._save_pair_item(job_id, item_id, index, prompt_obj, start, end, caption, replace)
                else:
                    prompt = prompt_override or prompt_obj.get("prompt") or ""
                    image = generate_image_vertex(
                        {
                            "modelId": settings["imageModel"],
                            "prompt": prompt,
                            "referenceDataUrls": refs,
                            "aspectRatio": settings["aspectRatio"],
                            "imageSize": settings["imageSize"],
                        }
                    )
                    caption = prompt
                    if settings.get("useVisionCaption"):
                        try:
                            caption = generate_text_vertex(
                                {
                                    "modelId": settings["llmModel"],
                                    "userText": "Caption this image for LoRA training in one dense paragraph.",
                                    "imageDataUrls": [f"data:{image.get('mimeType','image/png')};base64,{image['data']}"],
                                    "temperature": 0.4,
                                }
                            )["text"]
                        except Exception:
                            pass
                    if trigger:
                        caption = f"{trigger} {caption}".strip()
                    return self._save_single_item(job_id, item_id, index, prompt_obj, prompt, image, caption, replace)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc)
                retryable = any(tok in msg for tok in ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "503", "502", "timeout"))
                with self._lock:
                    job = self._jobs[job_id]
                    self._append_log(
                        job,
                        "WARN",
                        f"Item attempt {attempt} failed: {msg[:200]}",
                        retryable=retryable,
                    )
                if not retryable or not settings.get("autoResume", True) or attempt >= 8:
                    raise
                delay = min(120, 10 * (2 ** min(attempt - 1, 4)))
                for _ in range(delay):
                    if self._stopped(job_id) and not replace:
                        raise RuntimeError("stopped")
                    time.sleep(1)
        raise last_err or RuntimeError("generation failed")

    def _save_single_item(
        self,
        job_id: str,
        item_id: str | None,
        index: int,
        prompt_obj: dict[str, Any],
        prompt: str,
        image: dict[str, Any],
        caption: str,
        replace: bool,
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            slug = job["characterSlug"]
            if not item_id:
                item_id = f"{index + 1:04d}"
            dataset = self.character_dir(slug) / "dataset"
            img_path = dataset / f"{item_id}.png"
            txt_path = dataset / f"{item_id}.txt"
            meta_path = dataset / f"{item_id}.json"
            raw = base64.b64decode(image["data"])
            img_path.write_bytes(raw)
            txt_path.write_text(caption, encoding="utf-8")
            item = {
                "id": item_id,
                "status": "ok",
                "mode": "single",
                "tag": prompt_obj.get("tag") or "",
                "prompt": prompt,
                "text": caption,
                "imageUrl": f"/api/files/{slug}/dataset/{item_id}.png",
                "textUrl": f"/api/files/{slug}/dataset/{item_id}.txt",
                "updatedAt": now_iso(),
                "error": "",
            }
            meta_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
            items = job.setdefault("items", [])
            existing = next((x for x in items if x.get("id") == item_id), None)
            if existing:
                existing.update(item)
            else:
                items.append(item)
                items.sort(key=lambda x: x.get("id") or "")
            if not replace:
                job["completed"] = sum(1 for x in items if x.get("status") == "ok")
                job["failed"] = sum(1 for x in items if x.get("status") == "failed")
                job["current"] = len(items)
            self._append_log(job, "INFO", f"#{item_id} saved")
            self._persist_job(job)
            return item

    def _save_pair_item(
        self,
        job_id: str,
        item_id: str | None,
        index: int,
        prompt_obj: dict[str, Any],
        start: dict[str, Any],
        end: dict[str, Any],
        caption: str,
        replace: bool,
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            slug = job["characterSlug"]
            if not item_id:
                item_id = f"{index + 1:04d}"
            dataset = self.character_dir(slug) / "dataset"
            (dataset / f"{item_id}_start.png").write_bytes(base64.b64decode(start["data"]))
            (dataset / f"{item_id}_end.png").write_bytes(base64.b64decode(end["data"]))
            (dataset / f"{item_id}.txt").write_text(caption, encoding="utf-8")
            item = {
                "id": item_id,
                "status": "ok",
                "mode": "pair",
                "tag": prompt_obj.get("action_name") or "",
                "prompt": prompt_obj.get("edit_prompt") or "",
                "basePrompt": prompt_obj.get("base_prompt") or "",
                "editPrompt": prompt_obj.get("edit_prompt") or "",
                "text": caption,
                "startUrl": f"/api/files/{slug}/dataset/{item_id}_start.png",
                "endUrl": f"/api/files/{slug}/dataset/{item_id}_end.png",
                "imageUrl": f"/api/files/{slug}/dataset/{item_id}_end.png",
                "textUrl": f"/api/files/{slug}/dataset/{item_id}.txt",
                "updatedAt": now_iso(),
                "error": "",
            }
            (dataset / f"{item_id}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
            items = job.setdefault("items", [])
            existing = next((x for x in items if x.get("id") == item_id), None)
            if existing:
                existing.update(item)
            else:
                items.append(item)
                items.sort(key=lambda x: x.get("id") or "")
            if not replace:
                job["completed"] = sum(1 for x in items if x.get("status") == "ok")
                job["failed"] = sum(1 for x in items if x.get("status") == "failed")
                job["current"] = len(items)
            self._append_log(job, "INFO", f"#{item_id} pair saved")
            self._persist_job(job)
            return item

    def _run_job(self, job_id: str) -> None:
        try:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "running"
                self._append_log(job, "INFO", "Job running on backend (continues if browser closes)")
                self._persist_job(job)

            with self._lock:
                job = self._jobs[job_id]
            self._append_log(job, "INFO", "Building prompts…")
            prompts = self._build_prompts(job)
            with self._lock:
                job = self._jobs[job_id]
                job["total"] = len(prompts)
                self._append_log(job, "INFO", f"Got {len(prompts)} prompts")
                self._persist_job(job)

            for i, prompt_obj in enumerate(prompts):
                if self._stopped(job_id):
                    with self._lock:
                        job = self._jobs[job_id]
                        job["status"] = "stopped"
                        self._append_log(job, "WARN", "Job stopped by user")
                        self._persist_job(job)
                    return
                item_id = f"{i + 1:04d}"
                with self._lock:
                    job = self._jobs[job_id]
                    self._append_log(job, "INFO", f"[{i + 1}/{len(prompts)}] generating {prompt_obj.get('tag') or ''}")
                try:
                    self._generate_one_item(job_id, item_id=item_id, prompt_obj=prompt_obj, index=i)
                except Exception as exc:  # noqa: BLE001
                    if str(exc) == "stopped":
                        with self._lock:
                            job = self._jobs[job_id]
                            job["status"] = "stopped"
                            self._append_log(job, "WARN", "Job stopped")
                            self._persist_job(job)
                        return
                    with self._lock:
                        job = self._jobs[job_id]
                        item = {
                            "id": item_id,
                            "status": "failed",
                            "mode": job["settings"]["mode"],
                            "tag": prompt_obj.get("tag") or prompt_obj.get("action_name") or "",
                            "prompt": prompt_obj.get("prompt") or prompt_obj.get("edit_prompt") or "",
                            "basePrompt": prompt_obj.get("base_prompt") or "",
                            "editPrompt": prompt_obj.get("edit_prompt") or "",
                            "text": "",
                            "error": str(exc),
                            "updatedAt": now_iso(),
                        }
                        items = job.setdefault("items", [])
                        existing = next((x for x in items if x.get("id") == item_id), None)
                        if existing:
                            existing.update(item)
                        else:
                            items.append(item)
                        job["failed"] = sum(1 for x in items if x.get("status") == "failed")
                        job["completed"] = sum(1 for x in items if x.get("status") == "ok")
                        job["current"] = i + 1
                        self._append_log(job, "ERROR", f"#{item_id} failed: {exc}")
                        self._persist_job(job)

            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "completed"
                job["completed"] = sum(1 for x in job.get("items", []) if x.get("status") == "ok")
                job["failed"] = sum(1 for x in job.get("items", []) if x.get("status") == "failed")
                self._append_log(
                    job,
                    "INFO",
                    f"Job complete: {job['completed']} ok, {job['failed']} failed",
                )
                self._persist_job(job)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job["status"] = "failed"
                    self._append_log(job, "ERROR", f"Job crashed: {exc}")
                    job["error"] = traceback.format_exc()[-2000:]
                    self._persist_job(job)
            self.log_fn("ERROR", "job.crash", jobId=job_id, error=str(exc))


# singleton filled by server.py
STORE: JobStore | None = None
