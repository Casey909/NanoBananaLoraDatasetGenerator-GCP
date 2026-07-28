/**
 * Frontend client — talks only to local server.py (Vertex ADC),
 * same pattern as onestopvideo browser → Python → google.genai Vertex.
 */

export const IMAGE_MODELS = {
  'gemini-3.1-flash-image': {
    id: 'gemini-3.1-flash-image',
    label: 'Nano Banana 2',
    sizes: ['0.5K', '1K', '2K', '4K'],
    maxRefs: 14,
  },
  'gemini-3.1-flash-lite-image': {
    id: 'gemini-3.1-flash-lite-image',
    label: 'Nano Banana 2 Lite',
    sizes: ['1K'],
    maxRefs: 4,
  },
  'gemini-3-pro-image': {
    id: 'gemini-3-pro-image',
    label: 'Nano Banana Pro',
    sizes: ['1K', '2K', '4K'],
    maxRefs: 14,
  },
};

export const LLM_MODELS = {
  'gemini-3.6-flash': { id: 'gemini-3.6-flash', label: 'Gemini 3.6 Flash' },
  'gemini-3.5-flash': { id: 'gemini-3.5-flash', label: 'Gemini 3.5 Flash' },
  'gemini-3.5-flash-lite': { id: 'gemini-3.5-flash-lite', label: 'Gemini 3.5 Flash Lite' },
};

let proxyHealth = null;

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
  try {
    const res = await fetch('/api/health', { cache: 'no-store' });
    if (!res.ok) {
      proxyHealth = { ok: false, detail: `HTTP ${res.status}`, checkedAt: Date.now() };
      return proxyHealth;
    }
    const json = await res.json();
    proxyHealth = { ...json, checkedAt: Date.now() };
    return proxyHealth;
  } catch (error) {
    proxyHealth = {
      ok: false,
      detail: `${error.message}. Start: python server.py`,
      checkedAt: Date.now(),
    };
    return proxyHealth;
  }
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

  let lastError;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = json?.error?.message || json?.message || `HTTP ${res.status}`;
        const err = new Error(msg);
        err.status = res.status;
        throw err;
      }
      return json;
    } catch (error) {
      lastError = error;
      const retryable =
        error.status === 429 ||
        error.status === 408 ||
        (error.status && error.status >= 500) ||
        /fetch|network|timeout|RESOURCE_EXHAUSTED|UNAVAILABLE/i.test(error.message || '');
      if (retryable && attempt < maxRetries) {
        await sleep(attempt * 2000);
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

  const json = await postJson('/api/generate-image', {
    modelId,
    prompt,
    referenceDataUrls: refs,
    aspectRatio,
    imageSize: size,
  });

  if (!json?.data) throw new Error('No image returned from server');
  const objectUrl = imageToObjectUrl(json);
  return {
    objectUrl,
    mimeType: json.mimeType || 'image/png',
    base64: json.data,
    text: json.text || '',
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
  const json = await postJson('/api/generate-text', {
    modelId,
    systemPrompt,
    userText,
    imageDataUrls,
    temperature,
    maxOutputTokens,
  });
  const text = (json?.text || '').trim();
  if (!text) throw new Error('Empty LLM response');
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
