/**
 * Frontend client — talks only to local server.py (Vertex ADC),
 * same pattern as onestopvideo browser → Python → google.genai Vertex.
 */

export const IMAGE_MODELS = {
  'gemini-3.1-flash-image': {
    id: 'gemini-3.1-flash-image',
    label: 'Nano Banana 2',
    sizes: ['1K', '2K', '4K'],
    maxRefs: 4,
  },
  'gemini-3.1-flash-lite-image': {
    id: 'gemini-3.1-flash-lite-image',
    label: 'Nano Banana 2 Lite',
    sizes: ['1K'],
    maxRefs: 2,
  },
  'gemini-3-pro-image': {
    id: 'gemini-3-pro-image',
    label: 'Nano Banana Pro',
    sizes: ['1K', '2K', '4K'],
    maxRefs: 4,
  },
};

export const LLM_MODELS = {
  'gemini-3.6-flash': { id: 'gemini-3.6-flash', label: 'Gemini 3.6 Flash' },
  'gemini-3.5-flash': { id: 'gemini-3.5-flash', label: 'Gemini 3.5 Flash' },
  'gemini-3.5-flash-lite': { id: 'gemini-3.5-flash-lite', label: 'Gemini 3.5 Flash Lite' },
};

let proxyHealth = null;
let debugSink = null; // optional (level, message, meta) => void
let clientLogSeq = 0;

export function setDebugSink(fn) {
  debugSink = typeof fn === 'function' ? fn : null;
}

function clog(level, message, meta = {}) {
  const entry = {
    ts: new Date().toISOString(),
    level,
    message,
    ...meta,
  };
  try {
    // Keep browser console useful too.
    const line = `[${level}] ${message}`;
    if (level === 'ERROR') console.error(line, meta);
    else if (level === 'WARN') console.warn(line, meta);
    else console.log(line, meta);
  } catch {
    /* ignore */
  }
  try {
    debugSink?.(level, message, entry);
  } catch {
    /* ignore */
  }
  return entry;
}

function nextClientRequestId(prefix = 'c') {
  clientLogSeq += 1;
  return `${prefix}-${Date.now().toString(36)}-${clientLogSeq}`;
}

export function getSettings() {
  return {
    provider: 'vertex-adc',
    apiKey: '',
    project: proxyHealth?.project || '',
    location: proxyHealth?.location || 'global',
    token: '',
    useProxy: true,
  };
}

export function saveSettings() {
  return getSettings();
}

export function clearSettings() {
  proxyHealth = null;
}

export async function refreshProxyHealth() {
  clog('DEBUG', 'health.check.start');
  try {
    const res = await fetch('/api/health', { cache: 'no-store' });
    if (!res.ok) {
      proxyHealth = { ok: false, detail: `HTTP ${res.status}`, checkedAt: Date.now() };
      clog('ERROR', 'health.check.fail', { status: res.status });
      return proxyHealth;
    }
    const json = await res.json();
    proxyHealth = { ...json, checkedAt: Date.now() };
    clog('INFO', 'health.check.ok', {
      project: json.project,
      location: json.location,
      detail: json.detail,
      logFile: json.logFile,
    });
    return proxyHealth;
  } catch (error) {
    proxyHealth = {
      ok: false,
      detail: `${error.message}. Start: python server.py`,
      checkedAt: Date.now(),
    };
    clog('ERROR', 'health.check.exception', { error: error.message });
    return proxyHealth;
  }
}

export async function fetchServerLogs(limit = 100) {
  const res = await fetch(`/api/logs?limit=${encodeURIComponent(String(limit))}`, { cache: 'no-store' });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json?.error?.message || `HTTP ${res.status}`);
  return json;
}

export function getProxyHealth() {
  return proxyHealth;
}

export function hasCredentials() {
  return Boolean(proxyHealth?.ok);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

export function imageToObjectUrl(image) {
  if (!image?.data) return null;
  const binary = atob(image.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: image.mimeType || 'image/png' });
  return URL.createObjectURL(blob);
}

async function postJson(path, payload, maxRetries = 5) {
  if (!proxyHealth) await refreshProxyHealth();
  if (!proxyHealth?.ok) {
    throw new Error(
      proxyHealth?.detail ||
        'Vertex ADC server not ready. On the host run: python server.py (uses gcloud ADC like onestopvideo)'
    );
  }

  const clientRequestId = payload.clientRequestId || nextClientRequestId('req');
  const body = { ...payload, clientRequestId };
  let lastError;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const t0 = performance.now();
    clog('DEBUG', 'api.request', {
      path,
      attempt,
      clientRequestId,
      keys: Object.keys(body).filter((k) => !/DataUrls|references/i.test(k)),
    });
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const json = await res.json().catch(() => ({}));
      const ms = Math.round(performance.now() - t0);
      if (!res.ok) {
        const msg = json?.error?.message || json?.message || `HTTP ${res.status}`;
        const reqId = json?.error?.reqId || json?.debug?.reqId || '';
        clog('ERROR', 'api.response_error', {
          path,
          attempt,
          status: res.status,
          ms,
          clientRequestId,
          reqId,
          error: msg.slice(0, 300),
        });
        const err = new Error(reqId ? `${msg} (reqId=${reqId})` : msg);
        err.status = res.status;
        err.reqId = reqId;
        throw err;
      }
      clog('INFO', 'api.response_ok', {
        path,
        attempt,
        ms,
        clientRequestId,
        reqId: json?.debug?.reqId || '',
        debug: json?.debug || null,
      });
      return json;
    } catch (error) {
      lastError = error;
      const retryable =
        error.status === 429 ||
        error.status === 408 ||
        (error.status && error.status >= 500) ||
        /fetch|network|timeout|RESOURCE_EXHAUSTED|UNAVAILABLE/i.test(error.message || '');
      if (retryable && attempt < maxRetries) {
        // Longer backoff on 429 / RESOURCE_EXHAUSTED.
        const base = /429|RESOURCE_EXHAUSTED/i.test(error.message || '') || error.status === 429
          ? 8000
          : 2000;
        const waitMs = attempt * base;
        clog('WARN', 'api.retry', {
          path,
          attempt,
          waitMs,
          clientRequestId,
          error: String(error.message || error).slice(0, 220),
        });
        await sleep(waitMs);
        continue;
      }
      throw error;
    }
  }
  throw lastError || new Error('Request failed');
}

export async function generateImage({
  modelId,
  prompt,
  referenceDataUrls = [],
  aspectRatio = '1:1',
  imageSize = '1K',
}) {
  const model = IMAGE_MODELS[modelId] || IMAGE_MODELS['gemini-3.1-flash-image'];
  const size = model.sizes.includes(imageSize) ? imageSize : model.sizes[0];
  const refs = referenceDataUrls.slice(0, model.maxRefs);
  const clientRequestId = nextClientRequestId('img');

  clog('INFO', 'generateImage.start', {
    clientRequestId,
    modelId,
    aspectRatio,
    imageSize: size,
    refCount: refs.length,
    promptPreview: String(prompt || '').slice(0, 120),
  });

  const json = await postJson('/api/generate-image', {
    modelId,
    prompt,
    referenceDataUrls: refs,
    aspectRatio,
    imageSize: size,
    clientRequestId,
  });

  if (!json?.data) throw new Error('No image returned from server');
  const objectUrl = imageToObjectUrl(json);
  clog('INFO', 'generateImage.ok', {
    clientRequestId,
    reqId: json?.debug?.reqId,
    ms: json?.debug?.ms,
    outChars: json.data.length,
  });
  return {
    objectUrl,
    mimeType: json.mimeType || 'image/png',
    base64: json.data,
    text: json.text || '',
    debug: json.debug || { clientRequestId },
  };
}

export async function generateText({
  modelId,
  systemPrompt = '',
  userText,
  imageDataUrls = [],
  temperature = 0.8,
  maxOutputTokens = 8192,
}) {
  const clientRequestId = nextClientRequestId('txt');
  clog('INFO', 'generateText.start', {
    clientRequestId,
    modelId,
    userPreview: String(userText || '').slice(0, 120),
    imageCount: imageDataUrls.length,
  });
  const json = await postJson('/api/generate-text', {
    modelId,
    systemPrompt,
    userText,
    imageDataUrls,
    temperature,
    maxOutputTokens,
    clientRequestId,
  });
  const text = (json?.text || '').trim();
  if (!text) throw new Error('Empty LLM response');
  clog('INFO', 'generateText.ok', {
    clientRequestId,
    reqId: json?.debug?.reqId,
    ms: json?.debug?.ms,
    outChars: text.length,
  });
  return text;
}

export function parseJsonArray(text) {
  const match = String(text).match(/\[[\s\S]*\]/);
  if (!match) throw new Error('Failed to parse JSON array from LLM response');
  return JSON.parse(match[0]);
}

export async function urlToBlob(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch image blob (${res.status})`);
  return res.blob();
}
