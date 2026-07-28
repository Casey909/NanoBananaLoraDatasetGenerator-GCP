#!/usr/bin/env python3
"""LoRA dataset local server — same Vertex ADC path as onestopvideo.

Uses google.genai Client(vertexai=True, project=..., location=...) with
Application Default Credentials (gcloud auth application-default login).
No API keys / pasted OAuth tokens in the browser.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "11904"))
BIND = os.environ.get("BIND", "0.0.0.0")
LOG_PATH = ROOT / "server.log"
_LOG_LOCK = threading.Lock()
_LOG_RING: deque[dict[str, Any]] = deque(maxlen=500)


def log(level: str, event: str, **fields: Any) -> None:
    """Structured debug log → stderr, server.log, and in-memory ring."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "level": level.upper(),
        "event": event,
        **fields,
    }
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _LOG_LOCK:
        _LOG_RING.append(entry)
        try:
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
    stream = sys.stderr if level.upper() in {"ERROR", "WARN"} else sys.stdout
    stream.write(line + "\n")
    stream.flush()


def _summarize_refs(refs: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, ref in enumerate(refs or []):
        raw = str(ref or "")
        mime = "unknown"
        raw_len = len(raw)
        if raw.startswith("data:") and "," in raw:
            header, b64 = raw.split(",", 1)
            mime = header[5:].split(";", 1)[0] if ";" in header else header[5:]
            raw_len = len(b64)
        out.append({"i": i, "mime": mime, "dataUrlChars": len(raw), "b64Chars": raw_len})
    return out


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Prefer onestopvideo .env when present (same project / Vertex settings).
_load_dotenv(Path(r"D:\Projects\onestopvideo\.env"))
_load_dotenv(ROOT / ".env")

PROJECT_ID = (
    os.environ.get("OSV_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
    or ""
).strip()
VERTEX_LOCATION = (
    os.environ.get("OSV_VERTEX_LOCATION")
    or os.environ.get("GOOGLE_CLOUD_LOCATION")
    or "global"
).strip() or "global"
VERTEX_IMAGE_LOCATION = (
    os.environ.get("OSV_VERTEX_IMAGE_LOCATION")
    or VERTEX_LOCATION
).strip() or "global"

STORE = None  # set in main()


def _client(location: str):
    from google import genai

    if not PROJECT_ID:
        raise RuntimeError(
            "Missing OSV_PROJECT_ID / GOOGLE_CLOUD_PROJECT. "
            "Set it or use onestopvideo .env"
        )
    return genai.Client(vertexai=True, project=PROJECT_ID, location=location)


def _normalize_mime(mime: str) -> str:
    m = (mime or "image/jpeg").strip().lower()
    if m in {"image/jpg", "image/pjpeg"}:
        return "image/jpeg"
    if m in {"image/x-png"}:
        return "image/png"
    if m.startswith("image/"):
        return m
    return "image/jpeg"


def _decode_data_url(value: str) -> tuple[str, bytes]:
    raw = (value or "").strip()
    if raw.startswith("data:") and "," in raw:
        header, b64 = raw.split(",", 1)
        mime = "image/jpeg"
        if ";" in header:
            mime = header[5:].split(";", 1)[0] or mime
        elif header.startswith("data:") and len(header) > 5:
            mime = header[5:] or mime
        return _normalize_mime(mime), base64.b64decode(b64)
    return "image/jpeg", base64.b64decode(raw)


def _prepare_ref_image(blob: bytes, mime: str, max_side: int = 1536) -> tuple[str, bytes]:
    """Resize/compress refs so Vertex doesn't 400 on huge phone photos."""
    mime = _normalize_mime(mime)
    try:
        from PIL import Image, ImageOps
        import io

        img = Image.open(io.BytesIO(blob))
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88, optimize=True)
        return "image/jpeg", out.getvalue()
    except Exception:
        # Fall back to original bytes with a safe mime.
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            mime = "image/jpeg"
        # Hard cap raw payload (~4MB) to avoid INVALID_ARGUMENT.
        if len(blob) > 4_000_000:
            raise ValueError(
                f"Reference image too large ({len(blob)} bytes) and could not be resized. "
                "Re-upload a smaller JPG/PNG."
            )
        return mime, blob


def _max_refs_for_model(model: str) -> int:
    # Match onestopvideo practical limit; lite is weaker with many refs.
    if "lite" in model:
        return 3
    return 4


def _build_image_config(types_mod: Any, aspect_ratio: str, image_size: str, allow_person: bool):
    # Vertex image models (same as onestopvideo): 1K/2K/4K only.
    size = image_size.strip().upper()
    if size == "0.5K":
        size = "1K"
    kwargs: dict[str, Any] = {"aspect_ratio": aspect_ratio or "1:1"}
    if allow_person:
        kwargs["person_generation"] = "ALLOW_ALL"
    if size in {"1K", "2K", "4K"}:
        kwargs["image_size"] = size
    try:
        return types_mod.ImageConfig(**kwargs)
    except Exception:
        try:
            return types_mod.ImageConfig(aspect_ratio=aspect_ratio or "1:1")
        except Exception:
            return None


def generate_image_vertex(payload: dict[str, Any], req_id: str = "") -> dict[str, Any]:
    from google.genai import types

    t0 = time.perf_counter()
    req_id = req_id or uuid.uuid4().hex[:10]
    model = (payload.get("modelId") or payload.get("model") or "").strip()
    prompt = (payload.get("prompt") or "").strip()
    aspect_ratio = (payload.get("aspectRatio") or "1:1").strip()
    image_size = (payload.get("imageSize") or "1K").strip().upper()
    refs = payload.get("referenceDataUrls") or payload.get("references") or []
    client_req = payload.get("clientRequestId") or ""

    if not model:
        raise ValueError("modelId is required")
    if not prompt:
        raise ValueError("prompt is required")

    log(
        "INFO",
        "image.start",
        reqId=req_id,
        clientRequestId=client_req,
        model=model,
        aspect=aspect_ratio,
        imageSize=image_size,
        promptChars=len(prompt),
        promptPreview=prompt[:160],
        refSummary=_summarize_refs(list(refs)),
        project=PROJECT_ID,
        location=VERTEX_IMAGE_LOCATION,
    )

    client = _client(VERTEX_IMAGE_LOCATION)
    max_refs = _max_refs_for_model(model)

    image_parts: list[Any] = []
    ref_sizes: list[int] = []
    for idx, ref in enumerate(list(refs)[:max_refs]):
        if not ref:
            continue
        mime_in, blob_in = _decode_data_url(str(ref))
        mime, blob = _prepare_ref_image(blob_in, mime_in)
        ref_sizes.append(len(blob))
        log(
            "DEBUG",
            "image.ref_prepared",
            reqId=req_id,
            index=idx,
            mimeIn=mime_in,
            mimeOut=mime,
            bytesIn=len(blob_in),
            bytesOut=len(blob),
        )
        image_parts.append(types.Part.from_bytes(data=blob, mime_type=mime))

    if image_parts:
        contents: Any = [
            types.Content(
                role="user",
                parts=[*image_parts, types.Part.from_text(text=prompt)],
            )
        ]
    else:
        contents = prompt

    def _call(allow_person: bool, size: str):
        config_kwargs: dict[str, Any] = {"response_modalities": ["TEXT", "IMAGE"]}
        image_cfg = _build_image_config(types, aspect_ratio, size, allow_person)
        if image_cfg is not None:
            config_kwargs["image_config"] = image_cfg
        log(
            "DEBUG",
            "image.vertex_call",
            reqId=req_id,
            model=model,
            allowPerson=allow_person,
            size=size,
            refCount=len(image_parts),
            preparedBytes=ref_sizes,
        )
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    try:
        response = _call(allow_person=True, size=image_size)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        log("WARN", "image.vertex_error", reqId=req_id, error=msg[:400])
        # Retry once with safer config on INVALID_ARGUMENT (common with phone refs).
        if "INVALID_ARGUMENT" in msg or "400" in msg:
            log("INFO", "image.retry_safer", reqId=req_id, reason="INVALID_ARGUMENT")
            try:
                response = _call(allow_person=False, size="1K")
            except Exception as exc2:
                log("ERROR", "image.retry_failed", reqId=req_id, error=str(exc2)[:400])
                raise RuntimeError(
                    "Vertex INVALID_ARGUMENT after ref sanitize. "
                    "Use 1–4 clear JPG/PNG face/body refs (not HEIC albums), "
                    f"parallel=1. Original: {msg[:240]}"
                ) from exc
        else:
            raise

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        log("ERROR", "image.no_candidates", reqId=req_id, feedback=str(feedback)[:300])
        raise RuntimeError(f"Vertex returned no image candidates ({feedback})")

    texts: list[str] = []
    images: list[dict[str, str]] = []
    parts = candidates[0].content.parts if candidates[0].content else []
    finish = getattr(candidates[0], "finish_reason", None)
    for part in parts or []:
        if getattr(part, "text", None) and not getattr(part, "thought", None):
            texts.append(part.text)
        idata = getattr(part, "inline_data", None)
        data = getattr(idata, "data", None) if idata else None
        if data:
            mime = getattr(idata, "mime_type", None) or "image/png"
            if isinstance(data, str):
                b64 = data
            else:
                b64 = base64.b64encode(data).decode("ascii")
            images.append({"mimeType": mime, "data": b64})

    if not images:
        log("ERROR", "image.no_bytes", reqId=req_id, finish=str(finish), textPreview=(texts[0][:200] if texts else ""))
        raise RuntimeError(
            "Vertex returned no image bytes"
            + (f": {texts[0][:200]}" if texts else "")
        )

    best = images[-1]
    ms = int((time.perf_counter() - t0) * 1000)
    log(
        "INFO",
        "image.ok",
        reqId=req_id,
        ms=ms,
        outMime=best["mimeType"],
        outChars=len(best["data"]),
        imageParts=len(images),
        finish=str(finish),
        textPreview=("\n".join(texts).strip()[:120]),
    )
    return {
        "mimeType": best["mimeType"],
        "data": best["data"],
        "text": "\n".join(texts).strip(),
        "debug": {"reqId": req_id, "ms": ms, "refBytes": ref_sizes, "finish": str(finish)},
    }


def generate_text_vertex(payload: dict[str, Any], req_id: str = "") -> dict[str, Any]:
    from google.genai import types

    t0 = time.perf_counter()
    req_id = req_id or uuid.uuid4().hex[:10]
    model = (payload.get("modelId") or payload.get("model") or "").strip()
    user_text = (payload.get("userText") or payload.get("prompt") or "").strip()
    system_prompt = (payload.get("systemPrompt") or "").strip()
    temperature = float(payload.get("temperature", 0.8))
    max_output_tokens = int(payload.get("maxOutputTokens", 8192))
    image_urls = payload.get("imageDataUrls") or []
    client_req = payload.get("clientRequestId") or ""

    if not model:
        raise ValueError("modelId is required")
    if not user_text:
        raise ValueError("userText is required")

    log(
        "INFO",
        "text.start",
        reqId=req_id,
        clientRequestId=client_req,
        model=model,
        temperature=temperature,
        maxOutputTokens=max_output_tokens,
        userChars=len(user_text),
        userPreview=user_text[:160],
        systemChars=len(system_prompt),
        imageCount=len([x for x in image_urls if x]),
        project=PROJECT_ID,
        location=VERTEX_LOCATION,
    )

    client = _client(VERTEX_LOCATION)

    parts: list[Any] = []
    for img in list(image_urls)[:4]:
        if not img:
            continue
        mime, blob = _decode_data_url(str(img))
        mime, blob = _prepare_ref_image(blob, mime, max_side=1280)
        if blob:
            parts.append(types.Part.from_bytes(data=blob, mime_type=mime))
    parts.append(types.Part.from_text(text=user_text))

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    try:
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as exc:  # noqa: BLE001
        log("ERROR", "text.vertex_error", reqId=req_id, error=str(exc)[:400])
        raise

    texts: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    if candidates and candidates[0].content:
        for part in candidates[0].content.parts or []:
            # Skip thought/reasoning parts when present.
            if getattr(part, "thought", None):
                continue
            if getattr(part, "text", None):
                texts.append(part.text)
    text = "\n".join(texts).strip() or (getattr(response, "text", None) or "").strip()
    if not text:
        log("ERROR", "text.empty", reqId=req_id, finish=str(finish))
        raise RuntimeError(f"Vertex returned empty text (finish={finish})")
    ms = int((time.perf_counter() - t0) * 1000)
    log("INFO", "text.ok", reqId=req_id, ms=ms, outChars=len(text), finish=str(finish), preview=text[:120])
    return {"text": text, "debug": {"reqId": req_id, "ms": ms, "finish": str(finish)}}


def parse_json_array_safe(text: str) -> list[Any]:
    match = re.search(r"\[[\s\S]*\]", str(text or ""))
    if not match:
        raise ValueError("Failed to parse JSON array from LLM response")
    return json.loads(match.group(0))


def health() -> dict[str, Any]:
    detail = "ok"
    ok = False
    try:
        # Touch ADC the same way onestopvideo does.
        _client(VERTEX_LOCATION)
        ok = True
        detail = f"vertex ADC project={PROJECT_ID} location={VERTEX_LOCATION}"
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        log("ERROR", "health.fail", error=detail[:300])
    return {
        "ok": ok,
        "proxy": True,
        "auth": "vertex-adc",
        "provider": "vertex",
        "project": PROJECT_ID,
        "location": VERTEX_LOCATION,
        "imageLocation": VERTEX_IMAGE_LOCATION,
        "detail": detail,
        "logFile": str(LOG_PATH),
        "logCount": len(_LOG_RING),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        log("DEBUG", "http.access", client=self.address_string(), message=(fmt % args))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        mime = "application/octet-stream"
        suf = path.suffix.lower()
        if suf in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suf == ".png":
            mime = "image/png"
        elif suf == ".webp":
            mime = "image/webp"
        elif suf == ".txt":
            mime = "text/plain; charset=utf-8"
        elif suf == ".json":
            mime = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/health":
            h = health()
            if STORE:
                h["jobsRunning"] = sum(1 for j in STORE.list_jobs() if j.get("status") in {"running", "queued", "stopping"})
                h["characters"] = len(STORE.list_characters())
            self._send_json(200, h)
            return
        if path == "/api/logs":
            try:
                limit = max(1, min(500, int((qs.get("limit") or ["100"])[0])))
            except Exception:
                limit = 100
            with _LOG_LOCK:
                items = list(_LOG_RING)[-limit:]
            self._send_json(200, {"ok": True, "count": len(items), "logs": items, "logFile": str(LOG_PATH)})
            return
        if path == "/api/characters":
            self._send_json(200, {"characters": STORE.list_characters() if STORE else []})
            return
        if path.startswith("/api/characters/") and path.count("/") == 3:
            slug = unquote(path.split("/")[3])
            try:
                self._send_json(200, STORE.get_character(slug))
            except Exception as exc:
                self._send_json(404, {"error": {"message": str(exc)}})
            return
        if path == "/api/jobs":
            slug = (qs.get("character") or [None])[0]
            self._send_json(200, {"jobs": STORE.list_jobs(slug) if STORE else []})
            return
        if path.startswith("/api/jobs/") and path.count("/") == 3:
            job_id = path.split("/")[3]
            job = STORE.get_job(job_id) if STORE else None
            if not job:
                self._send_json(404, {"error": {"message": "Job not found"}})
                return
            self._send_json(200, job)
            return
        if path.startswith("/api/files/"):
            # /api/files/{slug}/refs/face_front.jpg
            parts = path.split("/")
            # ['', 'api', 'files', slug, ...]
            if len(parts) < 5:
                self.send_error(404)
                return
            slug = unquote(parts[3])
            rel = "/".join(unquote(p) for p in parts[4:])
            try:
                self._send_file(STORE.resolve_file(slug, rel))
            except Exception as exc:
                self._send_json(404, {"error": {"message": str(exc)}})
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length") or "0")
        req_id = uuid.uuid4().hex[:10]
        t0 = time.perf_counter()
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            log("WARN", "http.bad_json", reqId=req_id, path=path, bytes=length)
            self._send_json(400, {"error": {"message": "Invalid JSON body", "reqId": req_id}})
            return

        log(
            "INFO",
            "http.request",
            reqId=req_id,
            path=path,
            bytes=length,
            client=self.address_string(),
            keys=sorted([k for k in payload.keys() if k not in {"referenceDataUrls", "imageDataUrls", "references", "dataUrl"}]),
        )

        try:
            # Character + job APIs
            if path == "/api/characters":
                meta = STORE.ensure_character(payload.get("name") or payload.get("characterName") or "")
                self._send_json(200, meta)
                return
            if path.startswith("/api/characters/") and path.endswith("/refs"):
                # POST /api/characters/{slug}/refs  {slot, dataUrl}
                slug = unquote(path.split("/")[3])
                slot = (payload.get("slot") or "").strip()
                data_url = payload.get("dataUrl") or ""
                url = STORE.save_ref_data_url(slug, slot, data_url)
                self._send_json(200, {"ok": True, "url": url, "slot": slot, "slug": slug})
                return
            if path == "/api/jobs":
                job = STORE.create_and_start_job(payload)
                self._send_json(200, job)
                return
            if path.startswith("/api/jobs/") and path.endswith("/stop"):
                job_id = path.split("/")[3]
                self._send_json(200, STORE.stop_job(job_id))
                return
            if "/items/" in path and path.endswith("/regenerate"):
                # /api/jobs/{id}/items/{itemId}/regenerate
                parts = path.split("/")
                job_id = parts[3]
                item_id = parts[5]
                job = STORE.regenerate_item(job_id, item_id, payload.get("prompt") or "")
                self._send_json(200, job)
                return

            if path == "/api/generate-image":
                result = generate_image_vertex(payload, req_id=req_id)
                log("INFO", "http.response", reqId=req_id, path=path, status=200, ms=int((time.perf_counter() - t0) * 1000))
                self._send_json(200, result)
                return
            if path == "/api/generate-text":
                result = generate_text_vertex(payload, req_id=req_id)
                log("INFO", "http.response", reqId=req_id, path=path, status=200, ms=int((time.perf_counter() - t0) * 1000))
                self._send_json(200, result)
                return
            if path == "/api/generateContent":
                if payload.get("prompt") is not None or payload.get("referenceDataUrls") is not None:
                    result = generate_image_vertex(payload, req_id=req_id)
                else:
                    result = generate_text_vertex(payload, req_id=req_id)
                self._send_json(200, result)
                return

            log("WARN", "http.not_found", reqId=req_id, path=path)
            self.send_error(404, "Not found")
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log(
                "ERROR",
                "http.handler_error",
                reqId=req_id,
                path=path,
                error=str(exc)[:500],
                traceback=tb[-1500:],
                ms=int((time.perf_counter() - t0) * 1000),
            )
            self._send_json(500, {"error": {"message": str(exc), "reqId": req_id}})


def main() -> None:
    global STORE
    from jobs import JobStore

    STORE = JobStore(log_fn=log)
    log(
        "INFO",
        "server.start",
        bind=BIND,
        port=PORT,
        root=str(ROOT),
        project=PROJECT_ID,
        location=VERTEX_LOCATION,
        imageLocation=VERTEX_IMAGE_LOCATION,
        logFile=str(LOG_PATH),
        dataRoot=str(STORE.data_root),
    )
    server = ThreadingHTTPServer((BIND, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("INFO", "server.stop")


if __name__ == "__main__":
    main()
