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
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "11904"))
BIND = os.environ.get("BIND", "0.0.0.0")


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


def _client(location: str):
    from google import genai

    if not PROJECT_ID:
        raise RuntimeError(
            "Missing OSV_PROJECT_ID / GOOGLE_CLOUD_PROJECT. "
            "Set it or use onestopvideo .env"
        )
    return genai.Client(vertexai=True, project=PROJECT_ID, location=location)


def _decode_data_url(value: str) -> tuple[str, bytes]:
    raw = (value or "").strip()
    if raw.startswith("data:") and "," in raw:
        header, b64 = raw.split(",", 1)
        mime = "image/png"
        if ";" in header:
            mime = header[5:].split(";", 1)[0] or mime
        return mime, base64.b64decode(b64)
    return "image/png", base64.b64decode(raw)


def generate_image_vertex(payload: dict[str, Any]) -> dict[str, Any]:
    from google.genai import types

    model = (payload.get("modelId") or payload.get("model") or "").strip()
    prompt = (payload.get("prompt") or "").strip()
    aspect_ratio = (payload.get("aspectRatio") or "1:1").strip()
    image_size = (payload.get("imageSize") or "1K").strip().upper()
    refs = payload.get("referenceDataUrls") or payload.get("references") or []

    if not model:
        raise ValueError("modelId is required")
    if not prompt:
        raise ValueError("prompt is required")

    client = _client(VERTEX_IMAGE_LOCATION)

    image_cfg_kwargs: dict[str, Any] = {
        "aspect_ratio": aspect_ratio,
        "person_generation": "ALLOW_ALL",
    }
    if image_size in {"0.5K", "1K", "2K", "4K"}:
        image_cfg_kwargs["image_size"] = image_size

    config_kwargs: dict[str, Any] = {"response_modalities": ["TEXT", "IMAGE"]}
    try:
        config_kwargs["image_config"] = types.ImageConfig(**image_cfg_kwargs)
    except Exception:
        try:
            config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)
        except Exception:
            pass

    image_parts: list[Any] = []
    for ref in list(refs)[:14]:
        if not ref:
            continue
        mime, blob = _decode_data_url(str(ref))
        if blob:
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

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"Vertex returned no image candidates ({feedback})")

    texts: list[str] = []
    images: list[dict[str, str]] = []
    parts = candidates[0].content.parts if candidates[0].content else []
    for part in parts or []:
        if getattr(part, "text", None):
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
        raise RuntimeError(
            "Vertex returned no image bytes"
            + (f": {texts[0][:200]}" if texts else "")
        )

    best = images[-1]
    return {
        "mimeType": best["mimeType"],
        "data": best["data"],
        "text": "\n".join(texts).strip(),
    }


def generate_text_vertex(payload: dict[str, Any]) -> dict[str, Any]:
    from google.genai import types

    model = (payload.get("modelId") or payload.get("model") or "").strip()
    user_text = (payload.get("userText") or payload.get("prompt") or "").strip()
    system_prompt = (payload.get("systemPrompt") or "").strip()
    temperature = float(payload.get("temperature", 0.8))
    max_output_tokens = int(payload.get("maxOutputTokens", 8192))
    image_urls = payload.get("imageDataUrls") or []

    if not model:
        raise ValueError("modelId is required")
    if not user_text:
        raise ValueError("userText is required")

    client = _client(VERTEX_LOCATION)

    parts: list[Any] = []
    for img in list(image_urls)[:8]:
        if not img:
            continue
        mime, blob = _decode_data_url(str(img))
        if blob:
            parts.append(types.Part.from_bytes(data=blob, mime_type=mime))
    parts.append(types.Part.from_text(text=user_text))

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(**config_kwargs),
    )

    texts: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates and candidates[0].content:
        for part in candidates[0].content.parts or []:
            # Skip thought/reasoning parts when present.
            if getattr(part, "thought", None):
                continue
            if getattr(part, "text", None):
                texts.append(part.text)
    text = "\n".join(texts).strip() or (getattr(response, "text", None) or "").strip()
    if not text:
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        raise RuntimeError(f"Vertex returned empty text (finish={finish})")
    return {"text": text}


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
    return {
        "ok": ok,
        "proxy": True,
        "auth": "vertex-adc",
        "provider": "vertex",
        "project": PROJECT_ID,
        "location": VERTEX_LOCATION,
        "imageLocation": VERTEX_IMAGE_LOCATION,
        "detail": detail,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/api/health":
            self._send_json(200, health())
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or "0")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "Invalid JSON body"}})
            return

        try:
            if path == "/api/generate-image":
                self._send_json(200, generate_image_vertex(payload))
                return
            if path == "/api/generate-text":
                self._send_json(200, generate_text_vertex(payload))
                return
            # Backward-compatible alias — still Vertex ADC, never browser keys.
            if path == "/api/generateContent":
                # If body looks like raw generateContent, reject with guidance.
                if "body" in payload and "prompt" not in payload:
                    self._send_json(
                        400,
                        {
                            "error": {
                                "message": "Use /api/generate-image or /api/generate-text (Vertex ADC)"
                            }
                        },
                    )
                    return
                if payload.get("prompt") is not None or payload.get("referenceDataUrls") is not None:
                    self._send_json(200, generate_image_vertex(payload))
                else:
                    self._send_json(200, generate_text_vertex(payload))
                return
            self.send_error(404, "Not found")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(500, {"error": {"message": str(exc)}})


def main() -> None:
    print(f"Serving {ROOT} on http://{BIND}:{PORT}", flush=True)
    print(f"Auth: Vertex ADC (same as onestopvideo)", flush=True)
    print(f"Project: {PROJECT_ID or '(missing)'}", flush=True)
    print(f"Vertex location: {VERTEX_LOCATION} / image: {VERTEX_IMAGE_LOCATION}", flush=True)
    server = ThreadingHTTPServer((BIND, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped", flush=True)


if __name__ == "__main__":
    main()
