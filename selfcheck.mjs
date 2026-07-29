/**
 * Offline self-check (no network).
 * Run: node selfcheck.mjs
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const load = (name) => readFileSync(join(root, name), 'utf8');

const gemini = await import(pathToFileURL(join(root, 'gemini.js')).href);
assert.ok(gemini.IMAGE_MODELS['gemini-3.1-flash-image']);
assert.ok(gemini.LLM_MODELS['gemini-3.6-flash']);

const server = load('server.py');
assert.match(server, /vertexai=True/);
assert.match(server, /\/api\/jobs/);
assert.match(server, /\/api\/characters/);
assert.match(server, /\/api\/files\//);
assert.match(server, /JobStore/);
assert.match(server, /_clamp_image_size/);
assert.match(server, /image\.size_clamped/);

const jobs = load('jobs.py');
assert.match(jobs, /class JobStore/);
assert.match(jobs, /"data"/);
assert.match(jobs, /"characters"/);
assert.match(jobs, /regenerate_item/);
assert.match(jobs, /autoResume/);

const shots = load('shots.py');
assert.match(shots, /CHARACTER_SHOT_TEMPLATES/);
assert.equal((shots.match(/"tag":/g) || []).length, 90);

const exportPy = load('ltx_export.py');
assert.match(exportPy, /768/);
assert.match(exportPy, /dataset\.json/);
assert.match(exportPy, /export_ltx_pack/);

const trainPy = load('ltx_train.py');
assert.match(trainPy, /LTX_TRAINER_ROOT/);
assert.match(trainPy, /start_local_train/);

const app = load('app.js');
assert.match(app, /beforeunload/);
assert.match(app, /startPolling/);
assert.match(app, /Regenerate/);
assert.match(app, /\/api\/jobs/);
assert.match(app, /openCropModal/);
assert.match(app, /exportLtxPack/);
assert.match(app, /trainLtxLocal/);

const html = load('index.html');
assert.match(html, /Start Backend Job/);
assert.match(html, /data\/characters/);
assert.match(html, /cropModal/);
assert.match(html, /Export LTX Train Pack/);
assert.match(html, /Train Locally/);

console.log('selfcheck OK');
