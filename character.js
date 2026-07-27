/**
 * Character LoRA shot presets — diversity packs for training consistency.
 */

export const REF_SLOTS = [
  { id: 'face_front', label: 'Face (front)', required: true },
  { id: 'face_side', label: 'Face (side / 3/4)', required: false },
  { id: 'body_full', label: 'Full body', required: false },
  { id: 'body_upper', label: 'Upper body', required: false },
  { id: 'extra', label: 'Extra ref (outfit / detail)', required: false },
];

/** Curated shot list used when Character LoRA mode generates prompts. */
export const CHARACTER_SHOT_TEMPLATES = [
  { tag: 'face_front', prompt: 'close-up front face portrait, eyes looking at camera, neutral expression, soft even studio lighting, plain light gray background, photorealistic, high detail skin and facial features' },
  { tag: 'face_three_quarter_left', prompt: 'three-quarter view face portrait angled left, soft smile, natural daylight, shallow depth of field, plain background' },
  { tag: 'face_three_quarter_right', prompt: 'three-quarter view face portrait angled right, calm expression, soft Rembrandt lighting, plain background' },
  { tag: 'face_profile_left', prompt: 'strict left profile headshot, neutral expression, clean silhouette, soft studio lighting, plain background' },
  { tag: 'face_profile_right', prompt: 'strict right profile headshot, neutral expression, clean silhouette, soft studio lighting, plain background' },
  { tag: 'face_looking_up', prompt: 'headshot looking slightly upward, soft hopeful expression, overhead soft light, plain background' },
  { tag: 'face_looking_down', prompt: 'headshot looking slightly downward, contemplative expression, soft side light, plain background' },
  { tag: 'expression_smile', prompt: 'close-up portrait with a natural genuine smile, eyes engaged, soft beauty lighting, plain background' },
  { tag: 'expression_serious', prompt: 'close-up portrait with a serious focused expression, cinematic soft key light, plain background' },
  { tag: 'expression_laugh', prompt: 'close-up portrait mid-laugh, joyful expression, natural outdoor light, soft bokeh background' },
  { tag: 'upper_body_front', prompt: 'waist-up portrait facing camera, relaxed arms, casual clothing, soft studio lighting, plain background' },
  { tag: 'upper_body_side', prompt: 'waist-up three-quarter pose, one hand visible, casual clothing, soft window light, plain background' },
  { tag: 'full_body_front', prompt: 'full body standing front view head to toe, natural stance, full outfit visible, even lighting, plain seamless background' },
  { tag: 'full_body_side', prompt: 'full body standing side view, natural posture, full outfit visible, even lighting, plain seamless background' },
  { tag: 'full_body_back', prompt: 'full body standing back view looking over shoulder toward camera, full outfit visible, even lighting, plain background' },
  { tag: 'sitting_pose', prompt: 'character sitting casually on a simple stool, waist-up framing, relaxed pose, soft studio lighting, plain background' },
  { tag: 'walking_pose', prompt: 'full body walking pose mid-stride, natural motion, outdoor soft daylight, simple blurred background' },
  { tag: 'hands_near_face', prompt: 'close portrait with one hand gently near the face/chin, elegant pose, soft beauty lighting, plain background' },
  { tag: 'different_outfit', prompt: 'waist-up portrait in a different casual outfit than the references, same person identity preserved, soft studio lighting, plain background' },
  { tag: 'outdoor_context', prompt: 'outdoor environmental portrait, upper body, natural daylight, park or street soft bokeh, identity consistent with references' },
  { tag: 'indoor_context', prompt: 'indoor lifestyle portrait, upper body, warm interior lighting, simple room background, identity consistent with references' },
  { tag: 'dramatic_light', prompt: 'dramatic cinematic portrait, strong contrast lighting, close face framing, dark gradient background, identity preserved' },
  { tag: 'soft_beauty', prompt: 'beauty headshot, soft diffused light, clean skin detail, gentle catchlights in eyes, light gray seamless background' },
  { tag: 'wide_angle_full', prompt: 'full body wide shot standing centered, head to toe visible, even lighting, plain seamless studio background' },
];

export function buildCharacterPromptSeed(characterName, triggerWord, extraNotes) {
  const name = characterName?.trim() || 'the character';
  const trigger = triggerWord?.trim();
  const notes = extraNotes?.trim();
  return {
    identityLock: [
      `Keep the exact same person / character identity as the reference images of ${name}.`,
      'Preserve face shape, eyes, nose, mouth, hair, skin tone, age, and distinctive features.',
      'Do not invent a different person. Photorealistic unless references are stylized.',
      notes ? `Additional notes: ${notes}` : '',
      trigger ? `If text captions are used later, the trigger word is "${trigger}".` : '',
    ]
      .filter(Boolean)
      .join(' '),
  };
}

export function selectShotTemplates(count) {
  const n = Math.max(1, Math.min(count, CHARACTER_SHOT_TEMPLATES.length));
  // Prefer curated order for LoRA diversity rather than random.
  return CHARACTER_SHOT_TEMPLATES.slice(0, n);
}

export function shotsToPromptObjects(shots, characterName, extraTheme) {
  const subject = characterName?.trim() || 'the character';
  const themeBit = extraTheme?.trim() ? ` Theme/context hint: ${extraTheme.trim()}.` : '';
  return shots.map((s) => ({
    prompt: `Using the provided reference images, generate a new photo of ${subject}: ${s.prompt}.${themeBit} Maintain identity consistency with all references.`,
    tag: s.tag,
  }));
}
