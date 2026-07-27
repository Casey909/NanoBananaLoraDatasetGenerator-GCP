/**
 * Offline self-check for gcp-lora-character-dataset (no network, no API keys).
 * Run: node selfcheck.mjs
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));

function load(name) {
  return readFileSync(join(root, name), 'utf8');
}

// Syntax / import surface
const gemini = await import(pathToFileURL(join(root, 'gemini.js')).href);
const character = await import(pathToFileURL(join(root, 'character.js')).href);

assert.ok(gemini.IMAGE_MODELS['gemini-3.1-flash-image']);
assert.ok(gemini.IMAGE_MODELS['gemini-3.1-flash-lite-image']);
assert.ok(gemini.IMAGE_MODELS['gemini-3-pro-image']);
assert.ok(gemini.LLM_MODELS['gemini-3.6-flash']);
assert.ok(gemini.LLM_MODELS['gemini-3.5-flash']);
assert.ok(gemini.LLM_MODELS['gemini-3.5-flash-lite']);

assert.equal(gemini.IMAGE_MODELS['gemini-3.1-flash-lite-image'].sizes.join(','), '1K');

const shots = character.selectShotTemplates(8);
assert.equal(shots.length, 8);
const prompts = character.shotsToPromptObjects(shots, 'Aria', 'casual wardrobe');
assert.equal(prompts.length, 8);
assert.match(prompts[0].prompt, /Aria/);
assert.match(prompts[0].prompt, /reference/i);

const seed = character.buildCharacterPromptSeed('Aria', 'ohwx', 'short bob');
assert.match(seed.identityLock, /Aria/);
assert.match(seed.identityLock, /ohwx/);
assert.match(seed.identityLock, /short bob/);

assert.deepEqual(
  character.REF_SLOTS.filter((s) => s.required).map((s) => s.id),
  ['face_front']
);

const sample = '{"candidates":[{"content":{"parts":[{"text":"hi"},{"inlineData":{"mimeType":"image/png","data":"aaaa"}},{"inlineData":{"mimeType":"image/png","data":"bbbb"}}]}}]}';
const parsed = JSON.parse(sample);
const parts = parsed.candidates[0].content.parts.filter((p) => p.inlineData?.data);
assert.equal(parts[parts.length - 1].inlineData.data, 'bbbb');

assert.throws(() => gemini.parseJsonArray('no json here'), /Failed to parse/);
assert.deepEqual(gemini.parseJsonArray('prefix [{"prompt":"x"}] suffix'), [{ prompt: 'x' }]);

const html = load('index.html');
assert.match(html, /type="module" src="app\.js"/);
assert.match(html, /Character LoRA/);
assert.doesNotMatch(html, /fal\.ai|@fal-ai/);

const app = load('app.js');
assert.match(app, /from '\.\/gemini\.js'/);
assert.match(app, /from '\.\/character\.js'/);
assert.doesNotMatch(app, /fal\.subscribe|@fal-ai\/client/);

const css = load('style.css');
assert.match(css, /--accent/);
assert.match(css, /\.character-slots/);

console.log('selfcheck OK');
