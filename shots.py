"""Canonical 90-shot recipe for LTX 2.3 character LoRA datasets."""

from __future__ import annotations

# Blocks must sum to 90: face 22 + hs 14 + upper 14 + full 14 + swim 14 + ctx 12
CHARACTER_SHOT_TEMPLATES: list[dict[str, str]] = [
    # ---- face identity (22) ----
    {"block": "face", "tag": "face_front", "prompt": "close-up front face portrait, eyes looking at camera, neutral expression, soft even studio lighting, plain light gray background, photorealistic, high detail skin and facial features"},
    {"block": "face", "tag": "face_front_soft_smile", "prompt": "close-up front face portrait, soft natural smile, eyes engaged, soft beauty lighting, plain background"},
    {"block": "face", "tag": "face_three_quarter_left", "prompt": "three-quarter view face portrait angled left, soft smile, natural daylight, shallow depth of field, plain background"},
    {"block": "face", "tag": "face_three_quarter_right", "prompt": "three-quarter view face portrait angled right, calm expression, soft Rembrandt lighting, plain background"},
    {"block": "face", "tag": "face_three_quarter_left_serious", "prompt": "three-quarter view face portrait angled left, serious focused expression, cinematic key light, plain background"},
    {"block": "face", "tag": "face_three_quarter_right_smile", "prompt": "three-quarter view face portrait angled right, gentle smile, warm window light, plain background"},
    {"block": "face", "tag": "face_profile_left", "prompt": "strict left profile headshot, neutral expression, clean silhouette, soft studio lighting, plain background"},
    {"block": "face", "tag": "face_profile_right", "prompt": "strict right profile headshot, neutral expression, clean silhouette, soft studio lighting, plain background"},
    {"block": "face", "tag": "face_looking_up", "prompt": "headshot looking slightly upward, soft hopeful expression, overhead soft light, plain background"},
    {"block": "face", "tag": "face_looking_down", "prompt": "headshot looking slightly downward, contemplative expression, soft side light, plain background"},
    {"block": "face", "tag": "expression_smile", "prompt": "close-up portrait with a natural genuine smile, eyes engaged, soft beauty lighting, plain background"},
    {"block": "face", "tag": "expression_serious", "prompt": "close-up portrait with a serious focused expression, cinematic soft key light, plain background"},
    {"block": "face", "tag": "expression_laugh", "prompt": "close-up portrait mid-laugh, joyful expression, natural outdoor light, soft bokeh background"},
    {"block": "face", "tag": "expression_surprised", "prompt": "close-up portrait with a mild surprised expression, raised brows, soft studio light, plain background"},
    {"block": "face", "tag": "expression_confident", "prompt": "close-up portrait with a confident subtle smirk, sharp jaw detail, soft rim light, plain background"},
    {"block": "face", "tag": "expression_soft_eyes", "prompt": "close-up portrait with soft relaxed eyes, calm mouth, diffused beauty lighting, plain background"},
    {"block": "face", "tag": "face_hard_light", "prompt": "close-up front portrait with hard directional light and defined shadows, high contrast, plain background"},
    {"block": "face", "tag": "face_golden_hour", "prompt": "close-up portrait in warm golden hour sunlight, gentle flare, natural skin texture, soft outdoor bokeh"},
    {"block": "face", "tag": "face_cool_light", "prompt": "close-up portrait under cool daylight, clean color, neutral expression, plain light background"},
    {"block": "face", "tag": "face_over_shoulder", "prompt": "close portrait looking back over one shoulder toward camera, three-quarter face, soft studio light, plain background"},
    {"block": "face", "tag": "face_chin_down", "prompt": "close-up portrait chin slightly down, eyes up to camera, beauty softbox lighting, plain background"},
    {"block": "face", "tag": "soft_beauty", "prompt": "beauty headshot, soft diffused light, clean skin detail, gentle catchlights in eyes, light gray seamless background"},
    # ---- head & shoulders (14) ----
    {"block": "head_shoulders", "tag": "hs_front_casual", "prompt": "head and shoulders portrait facing camera, casual top, relaxed posture, soft studio lighting, plain background"},
    {"block": "head_shoulders", "tag": "hs_three_quarter", "prompt": "head and shoulders three-quarter pose, casual clothing, soft window light, plain background"},
    {"block": "head_shoulders", "tag": "hs_profile_soft", "prompt": "head and shoulders left profile, clean neckline, soft studio light, plain background"},
    {"block": "head_shoulders", "tag": "hs_rim_light", "prompt": "head and shoulders portrait with rim light outlining hair and shoulders, dark gray gradient background"},
    {"block": "head_shoulders", "tag": "hs_hoodie", "prompt": "head and shoulders in a plain hoodie, identity preserved, soft even light, plain background"},
    {"block": "head_shoulders", "tag": "hs_jacket", "prompt": "head and shoulders wearing a simple jacket, collar visible, soft studio light, plain background"},
    {"block": "head_shoulders", "tag": "hs_tshirt", "prompt": "head and shoulders in a plain t-shirt, natural posture, soft daylight, plain background"},
    {"block": "head_shoulders", "tag": "hs_sweater", "prompt": "head and shoulders in a soft sweater, cozy look, warm indoor light, plain background"},
    {"block": "head_shoulders", "tag": "hs_looking_aside", "prompt": "head and shoulders looking slightly off-camera, thoughtful expression, soft side light, plain background"},
    {"block": "head_shoulders", "tag": "hs_hand_near_chin", "prompt": "head and shoulders with one hand near chin, elegant pose, soft beauty lighting, plain background"},
    {"block": "head_shoulders", "tag": "hs_wind_hair", "prompt": "head and shoulders with slight hair movement, outdoor breeze feel, natural daylight, soft bokeh"},
    {"block": "head_shoulders", "tag": "hs_dramatic", "prompt": "head and shoulders dramatic contrast lighting, cinematic mood, dark background, identity preserved"},
    {"block": "head_shoulders", "tag": "hs_bright_highkey", "prompt": "head and shoulders high-key bright lighting, clean minimal look, white seamless background"},
    {"block": "head_shoulders", "tag": "hs_neutral_passport", "prompt": "head and shoulders passport-style neutral face, even lighting, plain light gray background"},
    # ---- upper body clothed (14) ----
    {"block": "upper", "tag": "upper_body_front", "prompt": "waist-up portrait facing camera, relaxed arms, casual clothing, soft studio lighting, plain background"},
    {"block": "upper", "tag": "upper_body_side", "prompt": "waist-up three-quarter pose, one hand visible, casual clothing, soft window light, plain background"},
    {"block": "upper", "tag": "upper_arms_crossed", "prompt": "waist-up portrait arms lightly crossed, confident stance, casual outfit, soft studio light, plain background"},
    {"block": "upper", "tag": "upper_hands_pockets", "prompt": "waist-up portrait hands in pockets, relaxed casual outfit, outdoor soft daylight, simple background"},
    {"block": "upper", "tag": "upper_leaning", "prompt": "waist-up leaning slightly forward toward camera, engaged expression, soft key light, plain background"},
    {"block": "upper", "tag": "upper_sitting_stool", "prompt": "character sitting casually on a simple stool, waist-up framing, relaxed pose, soft studio lighting, plain background"},
    {"block": "upper", "tag": "upper_different_outfit", "prompt": "waist-up portrait in a different casual outfit than the references, same person identity preserved, soft studio lighting, plain background"},
    {"block": "upper", "tag": "upper_formal_shirt", "prompt": "waist-up in a simple button shirt, neat collar, soft beauty lighting, plain background"},
    {"block": "upper", "tag": "upper_athletic_top", "prompt": "waist-up in a fitted athletic top, sporty casual look, gym-soft lighting, plain background"},
    {"block": "upper", "tag": "upper_hands_gesture", "prompt": "waist-up mid conversation hand gesture, natural pose, casual clothing, soft indoor light, plain background"},
    {"block": "upper", "tag": "upper_backlit", "prompt": "waist-up backlit silhouette edges with face still readable, casual clothing, bright rear light, plain background"},
    {"block": "upper", "tag": "upper_side_light", "prompt": "waist-up strong side lighting, sculpted face and torso shape under clothing, plain dark gray background"},
    {"block": "upper", "tag": "hands_near_face", "prompt": "close portrait with one hand gently near the face/chin, elegant pose, soft beauty lighting, plain background"},
    {"block": "upper", "tag": "upper_looking_away", "prompt": "waist-up looking away from camera, profile-ish face readable, casual clothing, soft daylight, plain background"},
    # ---- full body clothed (14) ----
    {"block": "full", "tag": "full_body_front", "prompt": "full body standing front view head to toe, natural stance, full outfit visible, even lighting, plain seamless background"},
    {"block": "full", "tag": "full_body_side", "prompt": "full body standing side view, natural posture, full outfit visible, even lighting, plain seamless background"},
    {"block": "full", "tag": "full_body_back", "prompt": "full body standing back view looking over shoulder toward camera, full outfit visible, even lighting, plain background"},
    {"block": "full", "tag": "full_body_three_quarter", "prompt": "full body three-quarter standing pose, casual outfit, soft studio light, plain seamless background"},
    {"block": "full", "tag": "walking_pose", "prompt": "full body walking pose mid-stride, natural motion, outdoor soft daylight, simple blurred background"},
    {"block": "full", "tag": "full_body_sitting", "prompt": "full body sitting on a simple chair, relaxed legs, casual outfit, soft studio lighting, plain background"},
    {"block": "full", "tag": "full_body_crouch", "prompt": "full body slight crouch / ready stance, athletic casual clothes, even lighting, plain background"},
    {"block": "full", "tag": "wide_angle_full", "prompt": "full body wide shot standing centered, head to toe visible, even lighting, plain seamless studio background"},
    {"block": "full", "tag": "full_body_jeans_tee", "prompt": "full body in jeans and plain t-shirt, natural stance, soft daylight, plain background"},
    {"block": "full", "tag": "full_body_jacket_outfit", "prompt": "full body wearing jacket over casual clothes, standing relaxed, soft studio light, plain background"},
    {"block": "full", "tag": "full_body_contrapposto", "prompt": "full body contrapposto stance, weight on one leg, casual outfit, soft even light, plain background"},
    {"block": "full", "tag": "full_body_hands_hips", "prompt": "full body standing with hands on hips, confident pose, casual outfit, soft studio lighting, plain background"},
    {"block": "full", "tag": "full_body_outdoor_path", "prompt": "full body standing on a simple outdoor path, natural daylight, identity consistent, soft environmental bokeh"},
    {"block": "full", "tag": "full_body_studio_turn", "prompt": "full body slight turn to the right, head toward camera, casual outfit, seamless studio background"},
    # ---- shirtless / swimwear body lock (14) — tasteful, non-explicit ----
    {"block": "swim", "tag": "shirtless_studio_front", "prompt": "waist-up shirtless torso front view, tasteful fitness physique reference, soft studio lighting, plain background, non-explicit"},
    {"block": "swim", "tag": "shirtless_studio_three_quarter", "prompt": "waist-up shirtless three-quarter torso view, clear shoulder and chest proportions, soft studio light, plain background, non-explicit"},
    {"block": "swim", "tag": "shirtless_side", "prompt": "waist-up shirtless side profile torso, clean silhouette, soft even light, plain background, non-explicit"},
    {"block": "swim", "tag": "shirtless_outdoor", "prompt": "waist-up shirtless outdoors in soft daylight, natural skin tone, simple blurred background, tasteful non-explicit"},
    {"block": "swim", "tag": "swim_trunks_full_front", "prompt": "full body standing front view in plain swim trunks, beach or poolside, natural daylight, tasteful swimwear, non-explicit"},
    {"block": "swim", "tag": "swim_trunks_full_side", "prompt": "full body standing side view in plain swim trunks, poolside, soft daylight, tasteful swimwear, non-explicit"},
    {"block": "swim", "tag": "swim_trunks_three_quarter", "prompt": "full body three-quarter stance in plain swim trunks, beach sand, bright daylight, tasteful non-explicit"},
    {"block": "swim", "tag": "swimwear_sitting_edge", "prompt": "sitting on pool edge in plain swimwear, upper body and legs visible, bright daylight, tasteful non-explicit"},
    {"block": "swim", "tag": "rashguard_swim", "prompt": "full body in fitted rashguard and swim shorts, beach setting, natural daylight, athletic swim look, non-explicit"},
    {"block": "swim", "tag": "onepiece_or_modest_swim", "prompt": "full body in modest one-piece or gender-appropriate swimwear, poolside, soft daylight, tasteful non-explicit, identity preserved"},
    {"block": "swim", "tag": "bikini_or_swim_alt", "prompt": "full body in tasteful gender-appropriate swimwear alternative, beach, natural light, non-explicit, same person identity"},
    {"block": "swim", "tag": "shirtless_arms_relaxed", "prompt": "waist-up shirtless arms relaxed at sides, clear body proportions, soft beauty lighting, plain background, non-explicit"},
    {"block": "swim", "tag": "swim_walking_shore", "prompt": "full body walking along shoreline in swimwear, mid-stride, sunny daylight, tasteful non-explicit"},
    {"block": "swim", "tag": "towel_shoulder_shirtless", "prompt": "waist-up shirtless with towel draped over one shoulder, poolside, soft daylight, tasteful non-explicit"},
    # ---- context / hard cases (12) ----
    {"block": "context", "tag": "outdoor_context", "prompt": "outdoor environmental portrait, upper body, natural daylight, park or street soft bokeh, identity consistent with references"},
    {"block": "context", "tag": "indoor_context", "prompt": "indoor lifestyle portrait, upper body, warm interior lighting, simple room background, identity consistent with references"},
    {"block": "context", "tag": "dramatic_light", "prompt": "dramatic cinematic portrait, strong contrast lighting, close face framing, dark gradient background, identity preserved"},
    {"block": "context", "tag": "cafe_window", "prompt": "upper body seated by a cafe window, soft window light, casual clothes, identity consistent"},
    {"block": "context", "tag": "night_street_bokeh", "prompt": "upper body night street portrait, colorful bokeh lights, face clearly lit, identity preserved"},
    {"block": "context", "tag": "rain_soft", "prompt": "upper body soft rainy day portrait, damp hair tips, overcast light, identity consistent, plain-ish urban blur"},
    {"block": "context", "tag": "hat_partial_occlusion", "prompt": "close portrait wearing a simple hat, partial hair occlusion, face still clearly identifiable, soft studio light"},
    {"block": "context", "tag": "glasses_optional", "prompt": "close portrait wearing simple eyeglasses, eyes visible, soft beauty lighting, plain background, identity preserved"},
    {"block": "context", "tag": "hand_occlusion_face", "prompt": "close portrait with fingers lightly framing face without hiding key features, soft light, plain background"},
    {"block": "context", "tag": "windy_outdoor_full", "prompt": "full body outdoor windy day, clothes and hair slight motion, natural daylight, identity consistent"},
    {"block": "context", "tag": "gym_soft_context", "prompt": "upper body in simple gym setting, soft indoor light, athletic casual wear, identity preserved"},
    {"block": "context", "tag": "studio_color_gel", "prompt": "close portrait with subtle colored gel accent light, face clearly readable, identity preserved, dark seamless background"},
]


def validate_shot_recipe() -> None:
    if len(CHARACTER_SHOT_TEMPLATES) != 90:
        raise ValueError(f"Expected 90 shots, got {len(CHARACTER_SHOT_TEMPLATES)}")
    tags = [s["tag"] for s in CHARACTER_SHOT_TEMPLATES]
    if len(tags) != len(set(tags)):
        raise ValueError("Duplicate shot tags in recipe")
    counts: dict[str, int] = {}
    for s in CHARACTER_SHOT_TEMPLATES:
        counts[s["block"]] = counts.get(s["block"], 0) + 1
    expected = {
        "face": 22,
        "head_shoulders": 14,
        "upper": 14,
        "full": 14,
        "swim": 14,
        "context": 12,
    }
    if counts != expected:
        raise ValueError(f"Block counts mismatch: {counts} != {expected}")


validate_shot_recipe()
