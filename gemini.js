/**
 * Gemini / Vertex AI client for Nano Banana image models + Flash LLMs.
 * Supports Google AI Studio (API key) and Vertex AI (Bearer token).
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

const STORAGE = {
  provider: 'gcp_lora_provider',
  apiKey: 'gcp_lora_api_key',
  project: 'gcp_lora_project',
  location: 'gcp_lora_location',
  token: 'gcp_lora_access_token',
};

export function getSettings() {
  return {
    provider: localStorage.getItem(STORAGE.provider) || 'google-ai',
    apiKey: localStorage.getItem(STORAGE.apiKey) || '',
    project: localStorage.getItem(STORAGE.project) || '',
    location: localStorage.getItem(STORAGE.location) || 'global',
    token: localStorage.getItem(STORAGE.token) || '',
  };
}

export function saveSettings(partial) {
  const next = { ...getSettings(), ...partial };
  localStorage.setItem(STORAGE.provider, next.provider);
  localStorage.setItem(STORAGE.apiKey, next.apiKey || '');
  localStorage.setItem(STORAGE.project, next.project || '');
  localStorage.setItem(STORAGE.location, next.location || 'global');
  localStorage.setItem(STORAGE.token, next.token || '');
  return next;
}

export function clearSettings() {
  Object.values(STORAGE).forEach((k) => localStorage.removeItem(k));
}

export function hasCredentials() {
  const s = getSettings();
  if (s.provider === 'vertex') {
    return Boolean(s.project && s.token);
  }
  return Boolean(s.apiKey);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function stripDataUrl(dataUrlOrBase64) {
  if (!dataUrlOrBase64) return { mimeType: 'image/png', data: '' };
  const m = String(dataUrlOrBase64).match(/^data:([^;]+);base64,(.+)$/s);
  if (m) return { mimeType: m[1], data: m[2] };
  return { mimeType: 'image/png', data: dataUrlOrBase64 };
}

function buildEndpoint(modelId) {
  const s = getSettings();
  if (s.provider === 'vertex') {
    const loc = s.location || 'global';
    const host =
      loc === 'global'
        ? 'https://aiplatform.googleapis.com'
        : `https://${loc}-aiplatform.googleapis.com`;
    return `${host}/v1/projects/${encodeURIComponent(s.project)}/locations/${encodeURIComponent(loc)}/publishers/google/models/${encodeURIComponent(modelId)}:generateContent`;
  }
  return `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(modelId)}:generateContent?key=${encodeURIComponent(s.apiKey)}`;
}

function authHeaders() {
  const s = getSettings();
  const headers = { 'Content-Type': 'application/json' };
  if (s.provider === 'vertex') {
    headers.Authorization = `Bearer ${s.token}`;
  }
  return headers;
}

function extractText(responseJson) {
  const parts = responseJson?.candidates?.[0]?.content?.parts || [];
  return parts
    .map((p) => p.text)
    .filter(Boolean)
    .join('\n')
    .trim();
}

function extractImages(responseJson) {
  const parts = responseJson?.candidates?.[0]?.content?.parts || [];
  return parts
    .filter((p) => p.inlineData?.data || p.inline_data?.data)
    .map((p) => {
      const inline = p.inlineData || p.inline_data;
      return {
        mimeType: inline.mimeType || inline.mime_type || 'image/png',
        data: inline.data,
      };
    });
}

function pickBestImage(images) {
  if (!images.length) return null;
  // Vertex may return multiple sizes; last part is usually the requested resolution.
  return images[images.length - 1];
}

export function imageToObjectUrl(image) {
  if (!image?.data) return null;
  const binary = atob(image.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: image.mimeType || 'image/png' });
  return URL.createObjectURL(blob);
}

async function generateContent(modelId, body, maxRetries = 3) {
  if (!hasCredentials()) {
    throw new Error('Add Google AI / Vertex credentials first (🔑)');
  }

  const endpoint = buildEndpoint(modelId);
  let lastError;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(body),
      });

      const json = await res.json().catch(() => ({}));

      if (!res.ok) {
        const msg =
          json?.error?.message ||
          json?.message ||
          `HTTP ${res.status} ${res.statusText}`;
        const err = new Error(msg);
        err.status = res.status;
        throw err;
      }

      const block = json?.promptFeedback?.blockReason || json?.candidates?.[0]?.finishReason;
      if (block === 'SAFETY' || block === 'BLOCKLIST') {
        throw new Error(`Response blocked (${block})`);
      }

      return json;
    } catch (error) {
      lastError = error;
      const retryable =
        error.status === 429 ||
        error.status === 408 ||
        (error.status && error.status >= 500) ||
        /fetch|network|timeout/i.test(error.message || '');
      if (retryable && attempt < maxRetries) {
        await sleep(attempt * 2000);
        continue;
      }
      throw error;
    }
  }

  throw lastError || new Error('Request failed');
}

/**
 * Generate or edit an image with optional reference images (data URLs or base64).
 */
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

  const parts = [];
  for (const ref of refs) {
    const { mimeType, data } = stripDataUrl(ref);
    if (data) {
      // Proto JSON uses camelCase on REST (inlineData / mimeType).
      parts.push({ inlineData: { mimeType, data } });
    }
  }
  parts.push({ text: prompt });

  const body = {
    contents: [{ role: 'user', parts }],
    generationConfig: {
      responseModalities: ['TEXT', 'IMAGE'],
      imageConfig: {
        aspectRatio,
        imageSize: size,
      },
    },
  };

  const json = await generateContent(modelId, body);
  const images = extractImages(json);
  const best = pickBestImage(images);
  if (!best) {
    const text = extractText(json);
    throw new Error(text ? `No image returned: ${text.slice(0, 200)}` : 'No image returned');
  }

  const objectUrl = imageToObjectUrl(best);
  return {
    objectUrl,
    mimeType: best.mimeType,
    base64: best.data,
    text: extractText(json),
  };
}

/**
 * Text / multimodal LLM call (prompts + captions).
 */
export async function generateText({
  modelId,
  systemPrompt = '',
  userText,
  imageDataUrls = [],
  temperature = 0.8,
  maxOutputTokens = 8192,
}) {
  const parts = [];
  for (const img of imageDataUrls) {
    const { mimeType, data } = stripDataUrl(img);
    if (data) parts.push({ inlineData: { mimeType, data } });
  }
  parts.push({ text: userText });

  const body = {
    contents: [{ role: 'user', parts }],
    generationConfig: {
      temperature,
      maxOutputTokens,
    },
  };

  if (systemPrompt) {
    body.systemInstruction = { parts: [{ text: systemPrompt }] };
  }

  const json = await generateContent(modelId, body);
  const text = extractText(json);
  if (!text) throw new Error('Empty LLM response');
  return text;
}

export function parseJsonArray(text) {
  const match = String(text).match(/\[[\s\S]*\]/);
  if (!match) throw new Error('Failed to parse JSON array from LLM response');
  return JSON.parse(match[0]);
}

/** Resolve a displayable URL (object URL or data URL) into a Blob for ZIP. */
export async function urlToBlob(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch image blob (${res.status})`);
  return res.blob();
}
