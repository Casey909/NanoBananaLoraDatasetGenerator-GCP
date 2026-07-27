/**
 * GCP Nano Banana LoRA Character Dataset Generator
 * Revamp of lovisdotio/NanoBananaLoraDatasetGenerator — FAL replaced with Gemini/Vertex.
 */

import {
  IMAGE_MODELS,
  LLM_MODELS,
  getSettings,
  saveSettings,
  clearSettings,
  hasCredentials,
  generateImage,
  generateText,
  parseJsonArray,
  urlToBlob,
} from './gemini.js';

import {
  REF_SLOTS,
  buildCharacterPromptSeed,
  selectShotTemplates,
  shotsToPromptObjects,
} from './character.js';

const state = {
  isGenerating: false,
  pairs: [],
  pairCounter: 0,
  mode: 'character', // character | pair | single | reference | import-edit
  referenceImageBase64: null,
  characterRefs: {}, // slotId -> dataUrl
  importedImages: [],
};

const DEFAULT_SYSTEM_PROMPTS = {
  character: `You are a prompt engineer for character LoRA training datasets.
Generate diverse, identity-preserving shot prompts for the same character.
Vary pose, angle, expression, framing, lighting, and background while locking identity.`,
  pair: `You are a creative prompt engineer for AI image generation. Generate diverse, detailed prompts for creating training data.
RULES:
1. Each prompt must be unique and creative
2. base_prompt: Detailed description for generating the START image
3. edit_prompt: Instruction for transforming START → END image
4. action_name: Short identifier for this transformation type`,
  single: `You are a creative prompt engineer for AI image generation. Generate diverse, detailed prompts for style/aesthetic training data.
RULES:
1. Each prompt must be unique and creative
2. prompt: Detailed description capturing aesthetic, style, composition, lighting, and mood`,
  reference: `You are a creative prompt engineer for AI image generation. Generate diverse prompts for variations of a reference image.
RULES:
1. Keep the subject recognizable
2. Vary poses, angles, backgrounds, lighting, and contexts`,
  'import-edit': `You describe image transformations clearly and specifically for an image-editing model.`,
};

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function truncate(str, length) {
  if (!str) return '';
  return str.length > length ? `${str.substring(0, length)}...` : str;
}

// =============================================================================
// Credentials UI
// =============================================================================

function showApiKeyModal() {
  document.getElementById('apiKeyModal').classList.remove('hidden');
  const s = getSettings();
  document.getElementById('providerSelect').value = s.provider;
  document.getElementById('apiKeyInput').value = s.apiKey;
  document.getElementById('vertexProjectInput').value = s.project;
  document.getElementById('vertexLocationInput').value = s.location || 'global';
  document.getElementById('vertexTokenInput').value = s.token;
  syncProviderFields();
  document.getElementById('apiKeyInput').focus();
}

function hideApiKeyModal() {
  document.getElementById('apiKeyModal').classList.add('hidden');
}

function syncProviderFields() {
  const provider = document.getElementById('providerSelect').value;
  document.getElementById('googleAiFields').classList.toggle('hidden', provider !== 'google-ai');
  document.getElementById('vertexFields').classList.toggle('hidden', provider !== 'vertex');
}

function toggleKeyVisibility(inputId, iconId) {
  const input = document.getElementById(inputId);
  const icon = document.getElementById(iconId);
  if (input.type === 'password') {
    input.type = 'text';
    icon.textContent = 'Hide';
  } else {
    input.type = 'password';
    icon.textContent = 'Show';
  }
}

function saveApiKey() {
  const provider = document.getElementById('providerSelect').value;
  const apiKey = document.getElementById('apiKeyInput').value.trim();
  const project = document.getElementById('vertexProjectInput').value.trim();
  const location = document.getElementById('vertexLocationInput').value.trim() || 'global';
  const token = document.getElementById('vertexTokenInput').value.trim();

  if (provider === 'google-ai' && !apiKey) {
    alert('Enter a Google AI Studio API key');
    return;
  }
  if (provider === 'vertex' && (!project || !token)) {
    alert('Enter Vertex project ID and access token');
    return;
  }

  saveSettings({ provider, apiKey, project, location, token });
  hideApiKeyModal();
  updateStatus(true, provider === 'vertex' ? 'Vertex ready' : 'API key saved');
}

function clearApiKey() {
  if (!confirm('Clear stored credentials?')) return;
  clearSettings();
  document.getElementById('apiKeyInput').value = '';
  document.getElementById('vertexProjectInput').value = '';
  document.getElementById('vertexTokenInput').value = '';
  updateStatus(false, 'No credentials');
}

// =============================================================================
// Mode / UI
// =============================================================================

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  const sections = {
    transformationSection: false,
    actionNameSection: false,
    referenceUploadSection: false,
    characterRefsSection: false,
    importEditSection: false,
    themeSection: true,
    characterNameSection: false,
  };

  const numPairsGroup = document.getElementById('numPairs')?.closest('.form-group');
  if (numPairsGroup) numPairsGroup.classList.remove('hidden');

  if (mode === 'character') {
    sections.characterRefsSection = true;
    sections.characterNameSection = true;
    document.getElementById('pairOrImageLabel').textContent = 'Images';
    document.getElementById('countLabel').textContent = 'images in memory';
    document.getElementById('progressLabel').textContent = 'images';
  } else if (mode === 'pair') {
    sections.transformationSection = true;
    sections.actionNameSection = true;
    document.getElementById('pairOrImageLabel').textContent = 'Pairs';
    document.getElementById('countLabel').textContent = 'pairs in memory';
    document.getElementById('progressLabel').textContent = 'pairs';
  } else if (mode === 'single') {
    document.getElementById('pairOrImageLabel').textContent = 'Images';
    document.getElementById('countLabel').textContent = 'images in memory';
    document.getElementById('progressLabel').textContent = 'images';
  } else if (mode === 'reference') {
    sections.referenceUploadSection = true;
    document.getElementById('pairOrImageLabel').textContent = 'Images';
    document.getElementById('countLabel').textContent = 'images in memory';
    document.getElementById('progressLabel').textContent = 'images';
  } else if (mode === 'import-edit') {
    sections.importEditSection = true;
    sections.themeSection = false;
    if (numPairsGroup) numPairsGroup.classList.add('hidden');
    document.getElementById('pairOrImageLabel').textContent = 'Images';
    document.getElementById('countLabel').textContent = 'edited pairs in memory';
    document.getElementById('progressLabel').textContent = 'images';
  }

  Object.entries(sections).forEach(([id, show]) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hidden', !show);
  });

  updateCostEstimate();
  updateSystemPromptPlaceholder();
}

function updateSystemPromptPlaceholder() {
  const textarea = document.getElementById('customSystemPrompt');
  textarea.placeholder = DEFAULT_SYSTEM_PROMPTS[state.mode] || '';
}

function toggleSystemPrompt() {
  const section = document.getElementById('systemPromptSection');
  const icon = document.getElementById('systemPromptIcon');
  const isHidden = section.classList.contains('hidden');
  section.classList.toggle('hidden');
  icon.textContent = isHidden ? '▼' : '▶';
}

function resetSystemPrompt() {
  document.getElementById('customSystemPrompt').value = '';
}

function getSystemPrompt() {
  const custom = document.getElementById('customSystemPrompt').value.trim();
  return custom || DEFAULT_SYSTEM_PROMPTS[state.mode];
}

function getImageModelId() {
  return document.getElementById('imageModel')?.value || 'gemini-3.1-flash-image';
}

function getLlmModelId() {
  return document.getElementById('llmModel')?.value || 'gemini-3.5-flash';
}

function syncResolutionOptions() {
  const model = IMAGE_MODELS[getImageModelId()];
  const select = document.getElementById('resolution');
  const current = select.value;
  select.innerHTML = '';
  for (const size of model.sizes) {
    const opt = document.createElement('option');
    opt.value = size;
    opt.textContent = size;
    select.appendChild(opt);
  }
  select.value = model.sizes.includes(current) ? current : model.sizes[0];
  updateCostEstimate();
}

// =============================================================================
// Uploads
// =============================================================================

function handleReferenceUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    state.referenceImageBase64 = e.target.result;
    const preview = document.getElementById('referencePreview');
    const placeholder = document.getElementById('uploadPlaceholder');
    preview.src = e.target.result;
    preview.classList.remove('hidden');
    placeholder.classList.add('hidden');
    document.getElementById('clearRefBtn').style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function clearReference() {
  state.referenceImageBase64 = null;
  const preview = document.getElementById('referencePreview');
  preview.classList.add('hidden');
  preview.src = '';
  document.getElementById('uploadPlaceholder').classList.remove('hidden');
  document.getElementById('clearRefBtn').style.display = 'none';
  document.getElementById('referenceInput').value = '';
}

function handleCharacterRefUpload(slotId, event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    state.characterRefs[slotId] = e.target.result;
    const preview = document.getElementById(`char-preview-${slotId}`);
    const placeholder = document.getElementById(`char-ph-${slotId}`);
    if (preview) {
      preview.src = e.target.result;
      preview.classList.remove('hidden');
    }
    if (placeholder) placeholder.classList.add('hidden');
  };
  reader.readAsDataURL(file);
}

function clearCharacterRef(slotId) {
  delete state.characterRefs[slotId];
  const preview = document.getElementById(`char-preview-${slotId}`);
  const placeholder = document.getElementById(`char-ph-${slotId}`);
  const input = document.getElementById(`char-input-${slotId}`);
  if (preview) {
    preview.classList.add('hidden');
    preview.src = '';
  }
  if (placeholder) placeholder.classList.remove('hidden');
  if (input) input.value = '';
}

function getCharacterRefList() {
  // Prefer required/front slots first for model attention order.
  const order = REF_SLOTS.map((s) => s.id);
  return order.map((id) => state.characterRefs[id]).filter(Boolean);
}

function renderCharacterSlots() {
  const grid = document.getElementById('characterSlots');
  if (!grid) return;
  grid.innerHTML = '';
  for (const slot of REF_SLOTS) {
    const wrap = document.createElement('div');
    wrap.className = 'char-slot';
    wrap.innerHTML = `
      <label class="char-slot-label">${slot.label}${slot.required ? ' *' : ''}</label>
      <div class="upload-zone char-zone" id="char-zone-${slot.id}">
        <input type="file" accept="image/*" id="char-input-${slot.id}" hidden />
        <div class="upload-placeholder" id="char-ph-${slot.id}">
          <span class="upload-icon">＋</span>
          <span>Upload</span>
        </div>
        <img id="char-preview-${slot.id}" class="reference-preview hidden" alt="${slot.label}" />
      </div>
      <button type="button" class="btn btn-sm btn-secondary" data-clear="${slot.id}">Clear</button>
    `;
    grid.appendChild(wrap);

    const zone = wrap.querySelector(`#char-zone-${slot.id}`);
    const input = wrap.querySelector(`#char-input-${slot.id}`);
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => handleCharacterRefUpload(slot.id, e));
    wrap.querySelector(`[data-clear="${slot.id}"]`).addEventListener('click', (e) => {
      e.stopPropagation();
      clearCharacterRef(slot.id);
    });
    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith('image/')) {
        handleCharacterRefUpload(slot.id, { target: { files: [file] } });
      }
    });
  }
}

async function handleImportUpload(event) {
  const files = Array.from(event.target.files || []).filter((f) => f.type.startsWith('image/'));
  if (!files.length) {
    alert('No images found');
    return;
  }

  state.importedImages = [];
  const previewList = document.getElementById('importPreviewList');
  previewList.innerHTML = '';
  previewList.classList.remove('hidden');
  document.getElementById('clearImportBtn').style.display = 'block';
  document.getElementById('importPlaceholder').innerHTML = `<span>Loading ${files.length} images...</span>`;

  await Promise.all(
    files.map(
      (file) =>
        new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => {
            state.importedImages.push({ file, base64: e.target.result, name: file.name });
            const thumb = document.createElement('div');
            thumb.className = 'import-thumb';
            thumb.innerHTML = `<img src="${e.target.result}" alt="${file.name}" /><span>${file.name}</span>`;
            previewList.appendChild(thumb);
            resolve();
          };
          reader.onerror = () => resolve();
          reader.readAsDataURL(file);
        })
    )
  );

  document.getElementById('importPlaceholder').innerHTML = `<span>${state.importedImages.length} images loaded</span>`;
  updateCostEstimate();
}

function clearImportedImages() {
  state.importedImages = [];
  document.getElementById('importPreviewList').innerHTML = '';
  document.getElementById('importPreviewList').classList.add('hidden');
  document.getElementById('clearImportBtn').style.display = 'none';
  document.getElementById('importInput').value = '';
  document.getElementById('importPlaceholder').innerHTML = `<span class="upload-icon">📂</span><span>Select a folder of images</span>`;
  updateCostEstimate();
}

// =============================================================================
// Status / progress / results
// =============================================================================

function updateStatus(connected, message) {
  document.getElementById('statusDot').className = `status-dot ${connected ? 'connected' : 'error'}`;
  document.getElementById('statusText').textContent = message;
}

function updatePairCount() {
  document.getElementById('pairCount').textContent = state.pairs.length;
}

function updateCostEstimate() {
  const useVision = document.getElementById('useVisionCaption').checked;
  const el = document.getElementById('costEstimate');
  if (state.mode === 'import-edit') {
    const n = state.importedImages.length;
    el.textContent = n ? `~${n} image edit call(s)${useVision ? ' + captions' : ''}` : 'Import images first';
    return;
  }
  const num = parseInt(document.getElementById('numPairs').value, 10) || 20;
  const imagesPerItem = state.mode === 'pair' ? 2 : 1;
  el.textContent = `~${num * imagesPerItem} image call(s)${useVision ? ' + captions' : ''} + 1 LLM prompt call`;
}

function showLoading(show, message = 'Working...') {
  const loader = document.getElementById('loadingIndicator');
  loader.classList.toggle('hidden', !show);
  loader.querySelector('span').textContent = message;
}

function showProgress(show) {
  document.getElementById('progressPanel').classList.toggle('hidden', !show);
}

function updateProgress(current, total, status) {
  const percent = total > 0 ? (current / total) * 100 : 0;
  document.getElementById('progressFill').style.width = `${percent}%`;
  document.getElementById('progressCurrent').textContent = current;
  document.getElementById('progressTotal').textContent = total;
  document.getElementById('progressStatus').textContent = status;
}

function addProgressLog(message, type = 'info') {
  const log = document.getElementById('progressLog');
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  entry.textContent = message;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function clearProgressLog() {
  document.getElementById('progressLog').innerHTML = '';
}

function addResultCard(item) {
  const container = document.getElementById('results');
  const card = document.createElement('div');
  card.className = 'result-card';

  if (item.mode === 'pair' || (item.startUrl && item.endUrl)) {
    card.innerHTML = `
      <div class="result-header"><span class="result-id">#${item.id}</span></div>
      <div class="result-images">
        <div class="result-image"><span class="label">START</span><img src="${item.startUrl}" alt="start" /></div>
        <div class="result-image"><span class="label">END</span><img src="${item.endUrl}" alt="end" /></div>
      </div>
      <div class="result-caption">${escapeHtml(truncate(item.text, 160))}</div>
    `;
  } else {
    card.innerHTML = `
      <div class="result-header"><span class="result-id">#${item.id}</span>${item.tag ? `<span class="result-tag">${escapeHtml(item.tag)}</span>` : ''}</div>
      <div class="result-images single">
        <div class="result-image"><img src="${item.imageUrl}" alt="result" /></div>
      </div>
      <div class="result-caption">${escapeHtml(truncate(item.text, 160))}</div>
    `;
  }

  container.insertBefore(card, container.firstChild);
}

function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

// =============================================================================
// Generation helpers
// =============================================================================

async function captionImage(imageDataUrlOrObjectUrl, modelId) {
  // Prefer fetching blob and converting if object URL; for captions pass as data URL when possible.
  let dataUrl = imageDataUrlOrObjectUrl;
  if (imageDataUrlOrObjectUrl.startsWith('blob:')) {
    const blob = await urlToBlob(imageDataUrlOrObjectUrl);
    dataUrl = await blobToDataUrl(blob);
  }
  return generateText({
    modelId,
    systemPrompt: 'Only answer the question. No markdown.',
    userText:
      'Caption this image for a text-to-image LoRA. Describe subject appearance, clothing, pose, expression, background, lighting, colors, and style in one dense paragraph.',
    imageDataUrls: [dataUrl],
    temperature: 0.4,
  });
}

async function captionEditPair(startUrl, endUrl, modelId) {
  const startData = startUrl.startsWith('blob:') ? await blobToDataUrl(await urlToBlob(startUrl)) : startUrl;
  const endData = endUrl.startsWith('blob:') ? await blobToDataUrl(await urlToBlob(endUrl)) : endUrl;
  return generateText({
    modelId,
    systemPrompt: 'Only describe the transformation in one sentence. No markdown.',
    userText:
      'Image 1 is BEFORE, image 2 is AFTER. Describe the edit/transformation applied.',
    imageDataUrls: [startData, endData],
    temperature: 0.3,
  });
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function generatePromptsWithLLM(theme, transformation, actionName, numPrompts, modelId) {
  const customSystemPrompt = getSystemPrompt();

  if (state.mode === 'pair') {
    const actionHint = actionName
      ? `Use this action name: "${actionName}"`
      : 'Generate a short descriptive action name';
    const userPrompt = `Generate ${numPrompts} unique prompt pairs for theme: "${theme}"
Transformation: "${transformation}"
${actionHint}

Return ONLY a valid JSON array:
[{"base_prompt":"...","edit_prompt":"...","action_name":"..."}]`;

    const text = await generateText({
      modelId,
      systemPrompt: `${customSystemPrompt}\n\n${actionHint}`,
      userText: userPrompt,
    });
    return parseJsonArray(text);
  }

  if (state.mode === 'single') {
    const userPrompt = `Generate ${numPrompts} unique image prompts for theme/style: "${theme}"
Return ONLY a valid JSON array:
[{"prompt":"..."}]`;
    const text = await generateText({
      modelId,
      systemPrompt: customSystemPrompt,
      userText: userPrompt,
    });
    return parseJsonArray(text);
  }

  if (state.mode === 'reference') {
    const userPrompt = `Generate ${numPrompts} unique variation prompts for: "${theme}"
Each prompt should describe a different scenario/pose/angle/background while keeping the subject consistent.
Return ONLY a valid JSON array:
[{"prompt":"..."}]`;
    const text = await generateText({
      modelId,
      systemPrompt: customSystemPrompt,
      userText: userPrompt,
    });
    return parseJsonArray(text);
  }

  if (state.mode === 'character') {
    const usePreset = document.getElementById('useCharacterPresets')?.checked !== false;
    const characterName = document.getElementById('characterName')?.value?.trim() || 'character';
    const triggerWord = document.getElementById('triggerWord')?.value?.trim() || '';

    if (usePreset) {
      const shots = selectShotTemplates(numPrompts);
      return shotsToPromptObjects(shots, characterName, theme);
    }

    const seed = buildCharacterPromptSeed(characterName, triggerWord, theme);
    const userPrompt = `Generate ${numPrompts} unique LoRA training shot prompts for character "${characterName}".
Theme/notes: "${theme}"
${seed.identityLock}

Cover face angles, expressions, upper body, full body, lighting, and contexts.
Return ONLY a valid JSON array:
[{"prompt":"...","tag":"short_tag"}]`;

    const text = await generateText({
      modelId,
      systemPrompt: customSystemPrompt,
      userText: userPrompt,
    });
    return parseJsonArray(text);
  }

  throw new Error(`Unsupported mode for LLM prompts: ${state.mode}`);
}

async function generatePairItem(prompt, index, total, aspectRatio, resolution, useVision, llmModel, triggerWord) {
  addProgressLog(`[${index + 1}/${total}] START: ${truncate(prompt.base_prompt, 40)}`, 'info');
  const start = await generateImage({
    modelId: getImageModelId(),
    prompt: prompt.base_prompt,
    aspectRatio,
    imageSize: resolution,
  });
  addProgressLog(`[${index + 1}] END edit...`, 'info');
  const end = await generateImage({
    modelId: getImageModelId(),
    prompt: prompt.edit_prompt,
    referenceDataUrls: [await blobToDataUrl(await urlToBlob(start.objectUrl))],
    aspectRatio,
    imageSize: resolution,
  });

  let finalText = prompt.action_name || prompt.edit_prompt;
  if (useVision) {
    try {
      finalText = await captionEditPair(start.objectUrl, end.objectUrl, llmModel);
    } catch (e) {
      console.warn('Vision caption failed', e);
    }
  }
  if (triggerWord) finalText = `${triggerWord} ${finalText}`;

  return {
    startUrl: start.objectUrl,
    endUrl: end.objectUrl,
    startPrompt: prompt.base_prompt,
    endPrompt: prompt.edit_prompt,
    actionName: prompt.action_name,
    text: finalText,
  };
}

async function generateSingleItem(prompt, index, total, aspectRatio, resolution, useVision, llmModel, triggerWord, refs = []) {
  addProgressLog(`[${index + 1}/${total}] ${truncate(prompt.prompt, 48)}`, 'info');
  const image = await generateImage({
    modelId: getImageModelId(),
    prompt: prompt.prompt,
    referenceDataUrls: refs,
    aspectRatio,
    imageSize: resolution,
  });

  let finalText = prompt.prompt;
  if (useVision) {
    try {
      finalText = await captionImage(image.objectUrl, llmModel);
    } catch (e) {
      console.warn('Vision caption failed', e);
    }
  }
  if (triggerWord) finalText = `${triggerWord} ${finalText}`;

  return {
    imageUrl: image.objectUrl,
    prompt: prompt.prompt,
    tag: prompt.tag || '',
    text: finalText,
  };
}

async function generateImportEditItem(importedImage, transformation, index, total, resolution, useVision, llmModel, triggerWord) {
  addProgressLog(`[${index + 1}/${total}] Editing ${importedImage.name}`, 'info');
  const end = await generateImage({
    modelId: getImageModelId(),
    prompt: transformation,
    referenceDataUrls: [importedImage.base64],
    aspectRatio: '1:1',
    imageSize: resolution,
  });

  let finalText = transformation;
  if (useVision) {
    try {
      finalText = await captionEditPair(importedImage.base64, end.objectUrl, llmModel);
    } catch (e) {
      console.warn('Vision caption failed', e);
    }
  }
  if (triggerWord) finalText = `${triggerWord} ${finalText}`;

  return {
    startUrl: importedImage.base64,
    endUrl: end.objectUrl,
    text: finalText,
    originalName: importedImage.name,
  };
}

// =============================================================================
// Main generation
// =============================================================================

async function startGeneration() {
  if (!hasCredentials()) {
    showApiKeyModal();
    return;
  }

  const numPairsInput = document.getElementById('numPairs');
  const numPairs = parseInt(numPairsInput.value, 10) || 20;
  const theme = document.getElementById('theme').value.trim();
  const transformation = document.getElementById('transformation').value.trim();
  const actionName = document.getElementById('actionName').value.trim();
  const triggerWord = document.getElementById('triggerWord').value.trim();
  const maxConcurrent = Math.max(1, Math.min(10, parseInt(document.getElementById('maxConcurrent')?.value, 10) || 2));
  const aspectRatio = document.getElementById('aspectRatio').value;
  const resolution = document.getElementById('resolution').value;
  const useVision = document.getElementById('useVisionCaption').checked;
  const llmModel = getLlmModelId();

  if (state.mode === 'import-edit') {
    const importTransformation = document.getElementById('importTransformation')?.value?.trim();
    if (!importTransformation) {
      alert('Describe the transformation to apply');
      return;
    }
    if (!state.importedImages.length) {
      alert('Import images first');
      return;
    }
  } else {
    if (numPairs > 40) {
      alert('Maximum 40 items per run. Run again to accumulate more.');
      numPairsInput.value = 40;
      return;
    }
    if (!theme && state.mode !== 'character') {
      alert('Fill in the dataset theme');
      return;
    }
    if (state.mode === 'character' && !theme) {
      // theme optional for character if name set — still require something
      const name = document.getElementById('characterName')?.value?.trim();
      if (!name) {
        alert('Enter a character name or theme/notes');
        return;
      }
    }
  }

  if (state.mode === 'pair' && !transformation) {
    alert('Fill in the transformation to learn');
    return;
  }
  if (state.mode === 'reference' && !state.referenceImageBase64) {
    alert('Upload a reference image');
    return;
  }
  if (state.mode === 'character') {
    if (!state.characterRefs.face_front) {
      alert('Upload at least a front face reference');
      return;
    }
  }

  if (state.mode === 'import-edit') {
    await runImportEdit(maxConcurrent, resolution, useVision, llmModel, triggerWord);
    return;
  }

  const currentMode = state.mode;
  const imagesPerItem = currentMode === 'pair' ? 2 : 1;
  const modeLabel = currentMode === 'pair' ? 'pairs' : 'images';
  if (
    !confirm(
      `Generate ${numPairs} ${modeLabel}?\nModel: ${getImageModelId()}\nLLM: ${llmModel}\nParallel: ${maxConcurrent}\n~${numPairs * imagesPerItem} image calls`
    )
  ) {
    return;
  }

  showProgress(true);
  clearProgressLog();
  updateProgress(0, numPairs, 'Generating prompts...');
  addProgressLog('Generating prompts...', 'info');
  state.isGenerating = true;

  let completed = 0;
  let failed = 0;

  try {
    const refs =
      currentMode === 'character'
        ? getCharacterRefList()
        : currentMode === 'reference'
          ? [state.referenceImageBase64]
          : [];

    if (currentMode === 'character') {
      addProgressLog(`Using ${refs.length} character reference image(s)`, 'info');
    }

    const prompts = await generatePromptsWithLLM(theme, transformation, actionName, numPairs, llmModel);
    addProgressLog(`Got ${prompts.length} prompts`, 'success');

    for (let i = 0; i < prompts.length; i += maxConcurrent) {
      if (!state.isGenerating) break;
      const batch = prompts.slice(i, Math.min(i + maxConcurrent, prompts.length));

      let results;
      if (currentMode === 'pair') {
        results = await Promise.allSettled(
          batch.map((p, bi) =>
            generatePairItem(p, i + bi, prompts.length, aspectRatio, resolution, useVision, llmModel, triggerWord)
          )
        );
      } else {
        results = await Promise.allSettled(
          batch.map((p, bi) =>
            generateSingleItem(p, i + bi, prompts.length, aspectRatio, resolution, useVision, llmModel, triggerWord, refs)
          )
        );
      }

      for (let j = 0; j < results.length; j++) {
        const result = results[j];
        if (result.status === 'fulfilled') {
          state.pairCounter += 1;
          const item = {
            id: String(state.pairCounter).padStart(4, '0'),
            mode: currentMode === 'pair' ? 'pair' : 'single',
            ...result.value,
          };
          state.pairs.push(item);
          addResultCard(item);
          updatePairCount();
          completed += 1;
          addProgressLog(`#${item.id} complete`, 'success');
        } else {
          failed += 1;
          addProgressLog(`${i + j + 1} failed: ${result.reason?.message || 'error'}`, 'error');
        }
        updateProgress(completed + failed, prompts.length, `${completed}/${prompts.length} done`);
      }
    }

    updateProgress(prompts.length, prompts.length, 'Complete');
    addProgressLog(`Done: ${completed} ok${failed ? `, ${failed} failed` : ''}`, 'success');
    addProgressLog('Click Download ZIP to save', 'info');
  } catch (error) {
    addProgressLog(`Error: ${error.message}`, 'error');
    alert(`Error: ${error.message}`);
  } finally {
    state.isGenerating = false;
  }
}

async function runImportEdit(maxConcurrent, resolution, useVision, llmModel, triggerWord) {
  const importTransformation = document.getElementById('importTransformation').value.trim();
  const imagesToProcess = [...state.importedImages];
  const totalImages = imagesToProcess.length;

  if (!confirm(`Edit ${totalImages} imported images with ${getImageModelId()}?`)) return;

  showProgress(true);
  clearProgressLog();
  updateProgress(0, totalImages, 'Starting edits...');
  state.isGenerating = true;

  let completed = 0;
  let failed = 0;

  try {
    for (let i = 0; i < imagesToProcess.length; i += maxConcurrent) {
      if (!state.isGenerating) break;
      const batch = imagesToProcess.slice(i, Math.min(i + maxConcurrent, totalImages));
      const results = await Promise.allSettled(
        batch.map((img, bi) =>
          generateImportEditItem(img, importTransformation, i + bi, totalImages, resolution, useVision, llmModel, triggerWord)
        )
      );

      for (const result of results) {
        if (result.status === 'fulfilled') {
          state.pairCounter += 1;
          const item = {
            id: String(state.pairCounter).padStart(4, '0'),
            mode: 'pair',
            ...result.value,
          };
          state.pairs.push(item);
          addResultCard(item);
          updatePairCount();
          completed += 1;
          addProgressLog(`#${item.id} ${result.value.originalName}`, 'success');
        } else {
          failed += 1;
          addProgressLog(`Failed: ${result.reason?.message || 'error'}`, 'error');
        }
        updateProgress(completed + failed, totalImages, `${completed}/${totalImages} done`);
      }
    }
    addProgressLog(`Done: ${completed} edited${failed ? `, ${failed} failed` : ''}`, 'success');
  } catch (error) {
    addProgressLog(`Error: ${error.message}`, 'error');
    alert(`Error: ${error.message}`);
  } finally {
    state.isGenerating = false;
  }
}

function stopGeneration() {
  state.isGenerating = false;
  addProgressLog('Stopped by user', 'info');
}

// =============================================================================
// ZIP
// =============================================================================

async function downloadZIP() {
  if (!state.pairs.length) {
    alert('No images to download');
    return;
  }

  const pairsSnapshot = [...state.pairs];
  showLoading(true, `Creating ZIP (${pairsSnapshot.length})...`);

  try {
    const JSZip = (await import('https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm')).default;
    const zip = new JSZip();

    for (let i = 0; i < pairsSnapshot.length; i++) {
      const item = pairsSnapshot[i];
      showLoading(true, `Adding ${i + 1}/${pairsSnapshot.length}...`);

      if (item.mode === 'pair' || (item.startUrl && item.endUrl)) {
        zip.file(`${item.id}_start.png`, await urlToBlob(item.startUrl));
        zip.file(`${item.id}_end.png`, await urlToBlob(item.endUrl));
        zip.file(`${item.id}.txt`, item.text || '');
      } else {
        zip.file(`${item.id}.png`, await urlToBlob(item.imageUrl));
        zip.file(`${item.id}.txt`, item.text || '');
      }
    }

    const content = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(content);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gcp_lora_dataset_${Date.now()}.zip`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    alert(`ZIP error: ${error.message}`);
  } finally {
    showLoading(false);
  }
}

function clearResults() {
  if (!state.pairs.length) return;
  if (!confirm(`Clear all ${state.pairs.length} items?`)) return;
  state.pairs = [];
  state.pairCounter = 0;
  document.getElementById('results').innerHTML = '';
  updatePairCount();
}

// =============================================================================
// Init
// =============================================================================

function populateModelSelects() {
  const imageSelect = document.getElementById('imageModel');
  imageSelect.innerHTML = '';
  Object.values(IMAGE_MODELS).forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.label} (${m.id})`;
    imageSelect.appendChild(opt);
  });
  imageSelect.value = 'gemini-3.1-flash-image';

  const llmSelect = document.getElementById('llmModel');
  llmSelect.innerHTML = '';
  Object.values(LLM_MODELS).forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.label} (${m.id})`;
    llmSelect.appendChild(opt);
  });
  llmSelect.value = 'gemini-3.5-flash';
}

function init() {
  populateModelSelects();
  syncResolutionOptions();
  renderCharacterSlots();

  if (hasCredentials()) {
    const s = getSettings();
    updateStatus(true, s.provider === 'vertex' ? 'Vertex ready' : 'API key set');
  } else {
    updateStatus(false, 'Click 🔑 to add credentials');
    setTimeout(() => showApiKeyModal(), 400);
  }

  document.getElementById('numPairs').addEventListener('input', updateCostEstimate);
  document.getElementById('useVisionCaption').addEventListener('change', updateCostEstimate);
  document.getElementById('resolution').addEventListener('change', updateCostEstimate);
  document.getElementById('imageModel').addEventListener('change', syncResolutionOptions);
  document.getElementById('providerSelect').addEventListener('change', syncProviderFields);

  updateCostEstimate();
  setMode('character');
  updatePairCount();

  const uploadZone = document.getElementById('uploadZone');
  if (uploadZone) {
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith('image/')) {
        handleReferenceUpload({ target: { files: [file] } });
      }
    });
  }
}

window.showApiKeyModal = showApiKeyModal;
window.hideApiKeyModal = hideApiKeyModal;
window.toggleKeyVisibility = toggleKeyVisibility;
window.saveApiKey = saveApiKey;
window.clearApiKey = clearApiKey;
window.startGeneration = startGeneration;
window.stopGeneration = stopGeneration;
window.downloadZIP = downloadZIP;
window.clearResults = clearResults;
window.setMode = setMode;
window.toggleSystemPrompt = toggleSystemPrompt;
window.resetSystemPrompt = resetSystemPrompt;
window.handleReferenceUpload = handleReferenceUpload;
window.clearReference = clearReference;
window.handleImportUpload = handleImportUpload;
window.clearImportedImages = clearImportedImages;
window.syncProviderFields = syncProviderFields;

document.addEventListener('DOMContentLoaded', init);
