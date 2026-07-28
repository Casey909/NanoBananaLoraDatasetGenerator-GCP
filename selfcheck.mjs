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

const jobs = load('jobs.py');
assert.match(jobs, /class JobStore/);
assert.match(jobs, /"data"/);
assert.match(jobs, /"characters"/);
assert.match(jobs, /regenerate_item/);
assert.match(jobs, /autoResume/);

const app = load('app.js');
assert.match(app, /beforeunload/);
assert.match(app, /startPolling/);
assert.match(app, /Regenerate/);
assert.match(app, /\/api\/jobs/);
assert.match(app, /openCropModal/);

const html = load('index.html');
assert.match(html, /Start Backend Job/);
assert.match(html, /data\/characters/);
assert.match(html, /cropModal/);

console.log('selfcheck OK');
