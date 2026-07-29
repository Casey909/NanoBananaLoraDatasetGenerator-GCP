/**
 * Character LoRA ref slots. Shot presets for generation live in shots.py (90 LTX recipe).
 */

export const REF_SLOTS = [
  { id: 'face_front', label: 'Face (front)', required: true },
  { id: 'face_side', label: 'Face (side / 3/4)', required: false },
  { id: 'body_full', label: 'Full body', required: false },
  { id: 'body_upper', label: 'Upper body', required: false },
  { id: 'extra', label: 'Extra ref (outfit / detail)', required: false },
];

/** @deprecated Shot list is owned by backend shots.py (90 templates). Kept for UI hints only. */
export const CHARACTER_SHOT_TEMPLATES = [
  { tag: 'face_front', prompt: 'close-up front face portrait' },
  { tag: 'swim_trunks_full_front', prompt: 'full body swim trunks (tasteful)' },
];

export const CHARACTER_SYSTEM = `You create diverse LoRA training shot prompts for one character.
Keep identity locked. Vary pose, framing, lighting, wardrobe, and context.
Include tasteful shirtless/swimwear body-proportion shots when asked for a full 90-pack.`;
