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
const character = await import(pathToFileURL(join(root, 'character.js')).href);

assert.ok(gemini.IMAGE_MODELS['gemini-3.1-flash-image']);
assert.ok(gemini.IMAGE_MODELS['gemini-3.1-flash-lite-image']);
assert.ok(gemini.IMAGE_MODELS['gemini-3-pro-image']);
assert.ok(gemini.LLM_MODELS['gemini-3.6-flash']);

const shots = character.selectShotTemplates(8);
assert.equal(shots.length, 8);

const app = load('app.js');
assert.match(app, /Vertex ADC/);
assert.doesNotMatch(app, /AIza|print-access-token|fal\.subscribe/);

const html = load('index.html');
assert.match(html, /application-default login/);
assert.doesNotMatch(html, /AI Studio API key|Access token/);

const server = load('server.py');
assert.match(server, /vertexai=True/);
assert.match(server, /genai\.Client/);
assert.match(server, /OSV_PROJECT_ID/);
assert.match(server, /\/api\/logs/);
assert.match(server, /def log\(/);
assert.doesNotMatch(server, /GEMINI_API_KEY|print-access-token/);

const geminiSrc = load('gemini.js');
assert.match(geminiSrc, /setDebugSink/);
assert.match(geminiSrc, /clientRequestId/);

assert.deepEqual(gemini.parseJsonArray('[{"prompt":"x"}]'), [{ prompt: 'x' }]);

console.log('selfcheck OK');
