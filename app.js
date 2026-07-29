/**
 * Frontend job client — generation runs on backend character folders.
 * Page can hide/sleep; poll resumes status/logs/results from disk-backed jobs.
 */

import {
  IMAGE_MODELS,
  LLM_MODELS,
  refreshProxyHealth,
  getProxyHealth,
  fetchServerLogs,
  setDebugSink,
} from './gemini.js';

import { REF_SLOTS } from './character.js';

const state = {
  characterSlug: localStorage.getItem('lora_character_slug') || '',
  characterName: localStorage.getItem('lora_character_name') || '',
  jobId: localStorage.getItem('lora_active_job') || '',
  pollTimer: null,
  lastLogCount: 0,
  refs: {}, // slot -> url
  job: null,
  dirtyCloseGuard: false,
};

function $(id) {
  return document.getElementById(id);
}

function truncate(str, n) {
  if (!str) return '';
  return str.length > n ? `${str.slice(0, n)}...` : str;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function updateStatus(ok, message) {
  $('statusDot').className = `status-dot ${ok ? 'connected' : 'error'}`;
  $('statusText').textContent = message;
}

function addProgressLog(message, type = 'info') {
  const log = $('progressLog');
  if (!log) return;
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  log.appendChild(entry);
  while (log.children.length > 500) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

function showProgress(show) {
  $('progressPanel')?.classList.toggle('hidden', !show);
}

function updateProgress(current, total, status) {
  const percent = total > 0 ? (current / total) * 100 : 0;
  $('progressFill').style.width = `${percent}%`;
  $('progressCurrent').textContent = current;
  $('progressTotal').textContent = total;
  $('progressStatus').textContent = status;
}

function setCloseGuard(on) {
  state.dirtyCloseGuard = !!on;
}

function jobIsActive(job) {
  return job && ['queued', 'running', 'stopping', 'regenerating'].includes(job.status);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json?.error?.message || `HTTP ${res.status}`);
  return json;
}

function fileToCompressedDataUrl(file, maxSide = 1536) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = reject;
    reader.onload = async () => {
      try {
        const rawUrl = reader.result;
        const img = await loadImage(rawUrl);
        resolve(compressImageElement(img, maxSide));
      } catch {
        resolve(reader.result);
      }
    };
    reader.readAsDataURL(file);
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = reject;
    el.src = src;
  });
}

function compressImageElement(img, maxSide = 1536, quality = 0.88) {
  const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(img, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', quality);
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = reject;
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });
}

/** Interactive crop → compressed JPEG data URL, or null if cancelled. */
function openCropModal(file, { title = 'Crop reference' } = {}) {
  return new Promise(async (resolve) => {
    const modal = $('cropModal');
    const stage = $('cropStage');
    const imgEl = $('cropImage');
    const box = $('cropBox');
    const aspectSel = $('cropAspect');
    if (!modal || !stage || !imgEl || !box) {
      // Fallback: no UI → compress full image
      resolve(await fileToCompressedDataUrl(file));
      return;
    }

    $('cropTitle').textContent = title;
    aspectSel.value = '1';
    const rawUrl = await readFileAsDataUrl(file);
    const natural = await loadImage(rawUrl);
    imgEl.src = rawUrl;

    // crop rect in displayed CSS pixels relative to stage
    const stateCrop = {
      x: 0,
      y: 0,
      w: 0,
      h: 0,
      drag: null, // { mode, startX, startY, orig }
      settled: false,
    };

    const aspectRatio = () => {
      const v = aspectSel.value;
      return v === 'free' ? null : Number(v);
    };

    const clampCrop = () => {
      const iw = imgEl.clientWidth;
      const ih = imgEl.clientHeight;
      const min = 40;
      stateCrop.w = Math.max(min, Math.min(stateCrop.w, iw));
      stateCrop.h = Math.max(min, Math.min(stateCrop.h, ih));
      const ar = aspectRatio();
      if (ar) {
        // keep aspect, prefer width then adjust height
        if (stateCrop.w / stateCrop.h > ar) {
          stateCrop.w = stateCrop.h * ar;
        } else {
          stateCrop.h = stateCrop.w / ar;
        }
        if (stateCrop.w > iw) {
          stateCrop.w = iw;
          stateCrop.h = iw / ar;
        }
        if (stateCrop.h > ih) {
          stateCrop.h = ih;
          stateCrop.w = ih * ar;
        }
      }
      stateCrop.x = Math.max(0, Math.min(stateCrop.x, iw - stateCrop.w));
      stateCrop.y = Math.max(0, Math.min(stateCrop.y, ih - stateCrop.h));
    };

    const paintBox = () => {
      clampCrop();
      box.style.left = `${stateCrop.x}px`;
      box.style.top = `${stateCrop.y}px`;
      box.style.width = `${stateCrop.w}px`;
      box.style.height = `${stateCrop.h}px`;
    };

    const initCrop = () => {
      const iw = imgEl.clientWidth;
      const ih = imgEl.clientHeight;
      if (!iw || !ih) return;
      const ar = aspectRatio() || 1;
      const side = Math.min(iw, ih) * 0.85;
      let w = side;
      let h = side / ar;
      if (h > ih * 0.9) {
        h = ih * 0.9;
        w = h * ar;
      }
      if (w > iw * 0.9) {
        w = iw * 0.9;
        h = w / (aspectRatio() || w / h);
      }
      stateCrop.w = w;
      stateCrop.h = h;
      stateCrop.x = (iw - w) / 2;
      stateCrop.y = (ih - h) / 2;
      paintBox();
      stateCrop.settled = true;
    };

    const finish = (dataUrl) => {
      cleanup();
      modal.classList.add('hidden');
      imgEl.removeAttribute('src');
      resolve(dataUrl);
    };

    const onPointerDown = (e) => {
      const handle = e.target?.dataset?.handle;
      const rect = stage.getBoundingClientRect();
      const clientX = e.clientX ?? e.touches?.[0]?.clientX;
      const clientY = e.clientY ?? e.touches?.[0]?.clientY;
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      stateCrop.drag = {
        mode: handle || 'move',
        startX: px,
        startY: py,
        orig: { x: stateCrop.x, y: stateCrop.y, w: stateCrop.w, h: stateCrop.h },
      };
      e.preventDefault();
      try {
        (e.currentTarget || box).setPointerCapture?.(e.pointerId);
      } catch {
        /* ignore */
      }
    };

    const onPointerMove = (e) => {
      if (!stateCrop.drag) return;
      const rect = stage.getBoundingClientRect();
      const clientX = e.clientX ?? e.touches?.[0]?.clientX;
      const clientY = e.clientY ?? e.touches?.[0]?.clientY;
      if (clientX == null) return;
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      const dx = px - stateCrop.drag.startX;
      const dy = py - stateCrop.drag.startY;
      const o = stateCrop.drag.orig;
      const mode = stateCrop.drag.mode;
      const ar = aspectRatio();
      const iw = imgEl.clientWidth;
      const ih = imgEl.clientHeight;

      if (mode === 'move') {
        stateCrop.x = o.x + dx;
        stateCrop.y = o.y + dy;
      } else {
        let x = o.x;
        let y = o.y;
        let w = o.w;
        let h = o.h;
        if (mode.includes('e')) w = o.w + dx;
        if (mode.includes('s')) h = o.h + dy;
        if (mode.includes('w')) {
          x = o.x + dx;
          w = o.w - dx;
        }
        if (mode.includes('n')) {
          y = o.y + dy;
          h = o.h - dy;
        }
        if (ar) {
          if (mode === 'n' || mode === 's') {
            w = h * ar;
            x = o.x + (o.w - w) / 2;
          } else if (mode === 'e' || mode === 'w') {
            h = w / ar;
            y = o.y + (o.h - h) / 2;
          } else if (mode === 'se' || mode === 'ne') {
            h = w / ar;
            if (mode === 'ne') y = o.y + o.h - h;
          } else if (mode === 'sw' || mode === 'nw') {
            w = Math.max(40, w);
            h = w / ar;
            x = o.x + o.w - w;
            if (mode === 'nw') y = o.y + o.h - h;
          }
        }
        stateCrop.x = x;
        stateCrop.y = y;
        stateCrop.w = Math.max(40, w);
        stateCrop.h = Math.max(40, h);
        if (stateCrop.x < 0) {
          stateCrop.w += stateCrop.x;
          stateCrop.x = 0;
        }
        if (stateCrop.y < 0) {
          stateCrop.h += stateCrop.y;
          stateCrop.y = 0;
        }
        if (stateCrop.x + stateCrop.w > iw) stateCrop.w = iw - stateCrop.x;
        if (stateCrop.y + stateCrop.h > ih) stateCrop.h = ih - stateCrop.y;
      }
      paintBox();
      e.preventDefault();
    };

    const onPointerUp = () => {
      stateCrop.drag = null;
    };

    const applyCrop = (full = false) => {
      const canvas = document.createElement('canvas');
      if (full) {
        canvas.width = natural.width;
        canvas.height = natural.height;
        canvas.getContext('2d').drawImage(natural, 0, 0);
      } else {
        const scaleX = natural.width / imgEl.clientWidth;
        const scaleY = natural.height / imgEl.clientHeight;
        const sx = Math.round(stateCrop.x * scaleX);
        const sy = Math.round(stateCrop.y * scaleY);
        const sw = Math.max(1, Math.round(stateCrop.w * scaleX));
        const sh = Math.max(1, Math.round(stateCrop.h * scaleY));
        canvas.width = sw;
        canvas.height = sh;
        canvas.getContext('2d').drawImage(natural, sx, sy, sw, sh, 0, 0, sw, sh);
      }
      // compress via temp image
      const tmp = new Image();
      tmp.onload = () => finish(compressImageElement(tmp, 1536));
      tmp.onerror = () => finish(canvas.toDataURL('image/jpeg', 0.88));
      tmp.src = canvas.toDataURL('image/jpeg', 0.92);
    };

    const onAspect = () => initCrop();
    const onReset = () => initCrop();
    const onCancel = () => finish(null);
    const onFull = () => applyCrop(true);
    const onApply = () => applyCrop(false);
    const onKey = (e) => {
      if (e.key === 'Escape') onCancel();
      if (e.key === 'Enter') onApply();
    };

    function cleanup() {
      box.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
      aspectSel.removeEventListener('change', onAspect);
      $('cropResetBtn')?.removeEventListener('click', onReset);
      $('cropCancelBtn')?.removeEventListener('click', onCancel);
      $('cropFullBtn')?.removeEventListener('click', onFull);
      $('cropApplyBtn')?.removeEventListener('click', onApply);
      window.removeEventListener('keydown', onKey);
    }

    box.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove, { passive: false });
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
    aspectSel.addEventListener('change', onAspect);
    $('cropResetBtn')?.addEventListener('click', onReset);
    $('cropCancelBtn')?.addEventListener('click', onCancel);
    $('cropFullBtn')?.addEventListener('click', onFull);
    $('cropApplyBtn')?.addEventListener('click', onApply);
    window.addEventListener('keydown', onKey);

    modal.classList.remove('hidden');
    // wait layout then init
    requestAnimationFrame(() => {
      if (imgEl.complete) initCrop();
      else imgEl.onload = () => initCrop();
    });
  });
}

function syncResolutionOptions() {
  const model = IMAGE_MODELS[$('imageModel')?.value] || IMAGE_MODELS['gemini-3.1-flash-image'];
  const res = $('resolution');
  if (!res || !model) return;
  const prev = res.value;
  res.innerHTML = '';
  (model.sizes || ['1K']).forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    res.appendChild(opt);
  });
  res.value = model.sizes.includes(prev) ? prev : model.sizes[0];
  const hint = $('resolutionHint');
  if (hint) {
    hint.textContent =
      model.id.includes('lite')
        ? 'Lite only supports 1K (2K/4K cause Vertex INVALID_ARGUMENT)'
        : `Allowed: ${model.sizes.join(', ')}`;
  }
}

function populateModelSelects() {
  const imageSelect = $('imageModel');
  imageSelect.innerHTML = '';
  Object.values(IMAGE_MODELS).forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.label} (${m.id})`;
    imageSelect.appendChild(opt);
  });
  imageSelect.value = 'gemini-3.1-flash-image';
  imageSelect.addEventListener('change', syncResolutionOptions);

  const llmSelect = $('llmModel');
  llmSelect.innerHTML = '';
  Object.values(LLM_MODELS).forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.label} (${m.id})`;
    llmSelect.appendChild(opt);
  });
  llmSelect.value = 'gemini-3.6-flash';

  syncResolutionOptions();
}

function renderCharacterSlots() {
  const grid = $('characterSlots');
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
          <span class="upload-icon">＋</span><span>Upload</span>
        </div>
        <img id="char-preview-${slot.id}" class="reference-preview hidden" alt="${slot.label}" />
      </div>
    `;
    grid.appendChild(wrap);
    const zone = wrap.querySelector(`#char-zone-${slot.id}`);
    const input = wrap.querySelector(`#char-input-${slot.id}`);
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      input.value = '';
      if (!file) return;
      await uploadRef(slot.id, file);
    });
  }
}

async function ensureCharacter() {
  const name = $('characterName').value.trim();
  if (!name) throw new Error('Enter a character name first');
  const meta = await api('/api/characters', {
    method: 'POST',
    body: JSON.stringify({ name, characterName: name }),
  });
  state.characterSlug = meta.slug;
  state.characterName = meta.name;
  localStorage.setItem('lora_character_slug', meta.slug);
  localStorage.setItem('lora_character_name', meta.name);
  $('characterSlugLabel').textContent = `Folder: data/characters/${meta.slug}/`;
  return meta;
}

async function uploadRef(slot, file) {
  if (!$('characterName').value.trim()) {
    alert('Enter a character name first');
    return;
  }
  showProgress(true);
  addProgressLog(`Crop ${slot}…`, 'info');
  let dataUrl;
  try {
    dataUrl = await openCropModal(file, { title: `Crop · ${slot}` });
  } catch (e) {
    addProgressLog(`Crop failed: ${e.message}`, 'error');
    return;
  }
  if (!dataUrl) {
    addProgressLog(`Crop cancelled for ${slot}`, 'warn');
    return;
  }
  addProgressLog(`Uploading ${slot} to character folder…`, 'info');
  await ensureCharacter();
  const result = await api(`/api/characters/${encodeURIComponent(state.characterSlug)}/refs`, {
    method: 'POST',
    body: JSON.stringify({ slot, dataUrl }),
  });
  state.refs[slot] = `${result.url}?t=${Date.now()}`;
  const preview = $(`char-preview-${slot}`);
  const ph = $(`char-ph-${slot}`);
  if (preview) {
    preview.src = state.refs[slot];
    preview.classList.remove('hidden');
  }
  if (ph) ph.classList.add('hidden');
  addProgressLog(`Saved ${slot} → ${result.url}`, 'success');
  setCloseGuard(true);
}

async function loadCharacterIntoUi(slug) {
  if (!slug) return;
  const data = await api(`/api/characters/${encodeURIComponent(slug)}`);
  state.characterSlug = data.slug;
  state.characterName = data.name;
  $('characterName').value = data.name;
  $('characterSlugLabel').textContent = `Folder: data/characters/${data.slug}/`;
  state.refs = data.refs || {};
  for (const slot of REF_SLOTS) {
    const preview = $(`char-preview-${slot.id}`);
    const ph = $(`char-ph-${slot.id}`);
    if (state.refs[slot.id] && preview) {
      preview.src = `${state.refs[slot.id]}?t=${Date.now()}`;
      preview.classList.remove('hidden');
      ph?.classList.add('hidden');
    }
  }
  renderResults(data.items || []);
  $('pairCount').textContent = (data.items || []).filter((x) => x.status === 'ok').length;

  // Attach latest job so Regenerate works after reload.
  try {
    const { jobs } = await api(`/api/jobs?character=${encodeURIComponent(slug)}`);
    if (jobs?.[0]?.id) {
      state.jobId = jobs[0].id;
      localStorage.setItem('lora_active_job', state.jobId);
      $('activeJobLabel').textContent = `Job ${jobs[0].id} (${jobs[0].status}) → ${slug}`;
    }
  } catch {
    /* ignore */
  }
}

function renderResults(items) {
  const container = $('results');
  container.innerHTML = '';
  const sorted = [...items].sort((a, b) => String(a.id).localeCompare(String(b.id)));
  for (const item of sorted) {
    container.appendChild(buildResultCard(item));
  }
}

function buildResultCard(item) {
  const card = document.createElement('div');
  card.className = 'result-card';
  card.dataset.itemId = item.id;
  const imgUrl = item.imageUrl || item.endUrl || '';
  const bust = item.updatedAt ? `?t=${encodeURIComponent(item.updatedAt)}` : `?t=${Date.now()}`;
  const statusClass = item.status === 'ok' ? 'success' : item.status === 'failed' ? 'error' : 'info';

  if (item.mode === 'pair' && item.startUrl && item.endUrl) {
    card.innerHTML = `
      <div class="result-header">
        <span class="result-id">#${escapeHtml(item.id)}</span>
        <span class="result-tag">${escapeHtml(item.status || '')}</span>
      </div>
      <div class="result-images">
        <div class="result-image"><span class="label">START</span><img src="${escapeHtml(item.startUrl + bust)}" alt="start" /></div>
        <div class="result-image"><span class="label">END</span><img src="${escapeHtml(item.endUrl + bust)}" alt="end" /></div>
      </div>
    `;
  } else {
    card.innerHTML = `
      <div class="result-header">
        <span class="result-id">#${escapeHtml(item.id)}</span>
        <span class="result-tag">${escapeHtml(item.tag || item.status || '')}</span>
      </div>
      <div class="result-images single">
        <div class="result-image">${imgUrl ? `<img src="${escapeHtml(imgUrl + bust)}" alt="result" />` : `<div class="log-${statusClass}" style="padding:12px">${escapeHtml(item.error || item.status || 'pending')}</div>`}</div>
      </div>
    `;
  }

  const refine = document.createElement('div');
  refine.className = 'refine-box';
  refine.innerHTML = `
    <label>Refine / regenerate prompt</label>
    <textarea class="refine-input" rows="3">${escapeHtml(item.prompt || item.editPrompt || '')}</textarea>
    <div class="refine-actions">
      <button type="button" class="btn btn-sm btn-primary regen-btn">Regenerate</button>
      <span class="refine-status"></span>
    </div>
  `;
  card.appendChild(refine);

  const caption = document.createElement('div');
  caption.className = 'result-caption';
  caption.textContent = truncate(item.text || item.error || '', 200);
  card.appendChild(caption);

  refine.querySelector('.regen-btn').addEventListener('click', async () => {
    if (!state.jobId) {
      alert('No active/loaded job. Start a generation job first (needed to track item ids).');
      return;
    }
    const prompt = refine.querySelector('.refine-input').value.trim();
    if (!prompt) {
      alert('Enter a refine prompt');
      return;
    }
    const statusEl = refine.querySelector('.refine-status');
    const btn = refine.querySelector('.regen-btn');
    btn.disabled = true;
    statusEl.textContent = 'queued…';
    try {
      setCloseGuard(true);
      await api(`/api/jobs/${encodeURIComponent(state.jobId)}/items/${encodeURIComponent(item.id)}/regenerate`, {
        method: 'POST',
        body: JSON.stringify({ prompt }),
      });
      statusEl.textContent = 'regenerating on backend…';
      addProgressLog(`Regenerate #${item.id} requested`, 'info');
      startPolling(state.jobId);
    } catch (e) {
      statusEl.textContent = e.message;
      addProgressLog(`Regenerate #${item.id} failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
    }
  });

  return card;
}

function applyJobToUi(job, { appendLogs = true } = {}) {
  state.job = job;
  const total = job.total || 0;
  const done = (job.completed || 0) + (job.failed || 0);
  updateProgress(Math.min(done, total) || job.current || 0, total || 1, `${job.status} · ${job.completed || 0} ok · ${job.failed || 0} failed`);
  $('pairCount').textContent = job.completed || 0;
  $('activeJobLabel').textContent = `Job ${job.id} (${job.status}) → ${job.characterSlug}`;

  if (appendLogs) {
    const logs = job.logs || [];
    if (logs.length > state.lastLogCount) {
      for (const row of logs.slice(state.lastLogCount)) {
        const type = row.level === 'ERROR' ? 'error' : row.level === 'WARN' ? 'warn' : 'info';
        addProgressLog(row.message, type);
      }
      state.lastLogCount = logs.length;
    }
  }

  renderResults(job.items || []);
  setCloseGuard(jobIsActive(job) || (job.items || []).some((x) => x.status === 'regenerating'));
  if (!jobIsActive(job) && !(job.items || []).some((x) => x.status === 'regenerating')) {
    stopPolling();
  }
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function startPolling(jobId) {
  state.jobId = jobId;
  localStorage.setItem('lora_active_job', jobId);
  stopPolling();
  showProgress(true);
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      applyJobToUi(job);
    } catch (e) {
      addProgressLog(`Poll error: ${e.message}`, 'error');
    }
  };
  tick();
  state.pollTimer = setInterval(tick, 1500);
}

async function startGeneration() {
  try {
    await ensureCharacter();
    if (!state.refs.face_front && !$('char-preview-face_front')?.src) {
      // re-check server
      const ch = await api(`/api/characters/${encodeURIComponent(state.characterSlug)}`);
      if (!ch.refs?.face_front) {
        alert('Upload face_front reference first (saved to character folder)');
        return;
      }
    }

    const payload = {
      characterName: state.characterName,
      characterSlug: state.characterSlug,
      mode: $('modeSelect')?.value || 'character',
      count: parseInt($('numPairs').value, 10) || 20,
      imageModel: $('imageModel').value,
      llmModel: $('llmModel').value,
      aspectRatio: $('aspectRatio').value,
      imageSize: $('resolution').value,
      triggerWord: $('triggerWord').value.trim(),
      theme: $('theme').value.trim(),
      useCharacterPresets: $('useCharacterPresets')?.checked !== false,
      useVisionCaption: $('useVisionCaption').checked,
      autoResume: $('autoResumeFailed')?.checked !== false,
      maxConcurrent: parseInt($('maxConcurrent').value, 10) || 1,
      transformation: $('transformation')?.value?.trim() || '',
    };

    if (!confirm(`Start backend job for "${state.characterName}"?\nImages save under data/characters/${state.characterSlug}/dataset/\nContinues even if you hide this page.`)) {
      return;
    }

    showProgress(true);
    state.lastLogCount = 0;
    $('progressLog').innerHTML = '';
    addProgressLog('Starting backend job…', 'info');
    const job = await api('/api/jobs', { method: 'POST', body: JSON.stringify(payload) });
    setCloseGuard(true);
    addProgressLog(`Job ${job.id} running on server`, 'success');
    startPolling(job.id);
  } catch (e) {
    alert(e.message);
    addProgressLog(e.message, 'error');
  }
}

async function stopGeneration() {
  if (!state.jobId) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(state.jobId)}/stop`, { method: 'POST', body: '{}' });
    addProgressLog('Stop requested', 'warn');
  } catch (e) {
    addProgressLog(`Stop failed: ${e.message}`, 'error');
  }
}

async function downloadZIP() {
  if (!state.characterSlug) {
    alert('No character selected');
    return;
  }
  // Simple: open character dataset listing via fetching items and zipping in browser from file URLs
  const ch = await api(`/api/characters/${encodeURIComponent(state.characterSlug)}`);
  const items = (ch.items || []).filter((x) => x.status === 'ok');
  if (!items.length) {
    alert('No saved images in character folder yet');
    return;
  }
  const JSZip = (await import('https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm')).default;
  const zip = new JSZip();
  for (const item of items) {
    if (item.mode === 'pair' && item.startUrl && item.endUrl) {
      zip.file(`${item.id}_start.png`, await (await fetch(item.startUrl)).blob());
      zip.file(`${item.id}_end.png`, await (await fetch(item.endUrl)).blob());
    } else if (item.imageUrl) {
      zip.file(`${item.id}.png`, await (await fetch(item.imageUrl)).blob());
    }
    if (item.textUrl) {
      zip.file(`${item.id}.txt`, await (await fetch(item.textUrl)).text());
    } else if (item.text) {
      zip.file(`${item.id}.txt`, item.text);
    }
  }
  const blob = await zip.generateAsync({ type: 'blob' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${state.characterSlug}_dataset_${Date.now()}.zip`;
  a.click();
}

async function downloadDebugLogs() {
  let serverLogs = [];
  try {
    const payload = await fetchServerLogs(300);
    serverLogs = payload.logs || [];
  } catch {
    /* ignore */
  }
  const blob = new Blob(
    [JSON.stringify({ job: state.job, characterSlug: state.characterSlug, serverLogs }, null, 2)],
    { type: 'application/json' }
  );
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `lora_debug_${Date.now()}.json`;
  a.click();
}

function syncModeUi() {
  const mode = $('modeSelect').value;
  $('transformationSection')?.classList.toggle('hidden', mode !== 'pair');
}

async function init() {
  setDebugSink(() => {});
  populateModelSelects();
  renderCharacterSlots();
  syncModeUi();
  $('modeSelect')?.addEventListener('change', syncModeUi);

  showProgress(true);
  addProgressLog('Backend job mode: generation continues if you hide the page', 'info');

  const health = await refreshProxyHealth();
  if (health?.ok) updateStatus(true, `Vertex ADC (${health.project || 'ready'})`);
  else updateStatus(false, health?.detail || 'Start python server.py');

  if (state.characterName) $('characterName').value = state.characterName;
  if (state.characterSlug) {
    try {
      await loadCharacterIntoUi(state.characterSlug);
    } catch (e) {
      addProgressLog(`Could not load character: ${e.message}`, 'warn');
    }
  }

  if (state.jobId) {
    addProgressLog(`Resuming active job ${state.jobId}…`, 'info');
    startPolling(state.jobId);
  }

  window.addEventListener('beforeunload', (e) => {
    if (!state.dirtyCloseGuard && !jobIsActive(state.job)) return;
    e.preventDefault();
    e.returnValue = 'Generation may still be running on the backend. Leave this page?';
    return e.returnValue;
  });

  // Visibility: keep polling; remind user backend continues.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && jobIsActive(state.job)) {
      console.log('[lora] page hidden — backend job still running', state.jobId);
    } else if (!document.hidden && state.jobId) {
      startPolling(state.jobId);
    }
  });
}

window.startGeneration = startGeneration;
window.stopGeneration = stopGeneration;
window.downloadZIP = downloadZIP;
window.downloadDebugLogs = downloadDebugLogs;
window.refreshAuthStatus = async () => {
  const h = await refreshProxyHealth();
  updateStatus(!!h?.ok, h?.ok ? `Vertex ADC (${h.project})` : h?.detail || 'not ready');
};
window.showApiKeyModal = () => {
  $('apiKeyModal')?.classList.remove('hidden');
  const h = getProxyHealth();
  $('authDetail').textContent = h?.ok
    ? `Vertex ADC ready — jobs save under data/characters/`
    : h?.detail || 'Start python server.py';
};
window.hideApiKeyModal = () => $('apiKeyModal')?.classList.add('hidden');
window.ensureCharacter = ensureCharacter;
window.refreshServerLogsIntoUi = async () => {
  try {
    const payload = await fetchServerLogs(30);
    for (const row of (payload.logs || []).slice(-15)) {
      addProgressLog(`[srv ${row.event}] ${truncate(JSON.stringify(row), 180)}`, row.level === 'ERROR' ? 'error' : 'info');
    }
  } catch (e) {
    addProgressLog(e.message, 'error');
  }
};

document.addEventListener('DOMContentLoaded', init);
