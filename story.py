"""
Story Engine Module for Dastaan Canvas
Handles Scene-by-Scene progression, narrative branching, character/palette lock,
and multi-lingual storytelling (English, Hindi, Hinglish).
"""

import json
import re
import requests
import streamlit as st
from emotion import get_api_key, EMOTION_CONFIG, CANDIDATE_MODELS

# Default archetypes for toy-story / magical realism character & setting lock
DEFAULT_CHARACTERS = [
    {
        "id": "babu_clockwork",
        "name": "Babu the Clockwork Sparrow",
        "description": "a handcrafted brass and porcelain mechanical sparrow with glowing sapphire eyes and miniature gears",
        "avatar": "🕊️"
    },
    {
        "id": "tara_starlight",
        "name": "Tara the Starlight Puppet",
        "description": "a nostalgic wooden marionette woven with luminescent indigo thread and golden embroidery",
        "avatar": "✨"
    },
    {
        "id": "kavi_cloudbear",
        "name": "Kavi the Cloud Bear",
        "description": "a fluffy velveteen toy bear wearing an aviator goggles and a small brass compass around its neck",
        "avatar": "🧸"
    }
]

DEFAULT_SETTINGS = [
    {
        "id": "grandpa_attic",
        "name": "Grandpa's Enchanted Attic",
        "description": "a cozy dusty attic filled with forgotten antique toys, grandfather clocks, dusty sunbeams, and floating amber dust motes"
    },
    {
        "id": "celestial_workshop",
        "name": "The Celestial Toymaker's Workshop",
        "description": "a magical workshop overlooking a starlit horizon, filled with floating miniature planets and luminescent paint vials"
    },
    {
        "id": "rainy_windowpane",
        "name": "The Rainy Windowpane Sanctuary",
        "description": "a quiet wooden windowsill looking out into gentle twilight rain, with warm reflection puddles and soft candle glow"
    }
]

class StoryEngine:
    @staticmethod
    def initialize_story_state():
        """Sets up persistent session state for narrative progression."""
        if "story_initialized" not in st.session_state:
            st.session_state["story_initialized"] = True
            st.session_state["current_scene_index"] = 0
            st.session_state["character"] = DEFAULT_CHARACTERS[0]
            st.session_state["setting"] = DEFAULT_SETTINGS[0]
            st.session_state["story_history"] = []
            st.session_state["language"] = "English"  # "English", "Hindi", "Hinglish"
            st.session_state["locked_palette"] = ["#FFB703", "#023047", "#8ECAE6"]

    @classmethod
    def generate_next_scene(
        cls,
        user_input: str,
        detected_emotion: dict,
        chosen_branch: str = None,
        language: str = "English"
    ) -> dict:
        """
        Generates the next chapter scene based on user emotion, chosen branch, and locked character.
        [BEHAVIOUR EXPRESSION]: The story pacing, vocabulary, and obstacle difficulty morph
        directly according to the user's emotional valence and arousal.
        """
        api_key = get_api_key()
        character = st.session_state.get("character", DEFAULT_CHARACTERS[0])
        setting = st.session_state.get("setting", DEFAULT_SETTINGS[0])
        scene_idx = len(st.session_state.get("story_history", [])) + 1
        emotion_name = detected_emotion.get("emotion", "neutral")
        valence = detected_emotion.get("valence", 0.0)
        arousal = detected_emotion.get("arousal", 0.5)

        cfg = EMOTION_CONFIG.get(emotion_name, EMOTION_CONFIG["neutral"])
        palette_str = ", ".join(cfg.get("palette", []))
        brush_texture = cfg.get("brush_texture", "cinematic soft rendering")
        camera_style = cfg.get("camera_style", "wide cinematic angle")
        story_tone = cfg.get("story_tone", "cinematic and evocative")

        # Context summary of past scenes
        past_scenes_summary = " ".join([
            f"Scene {s['scene_number']}: {s.get('short_summary', '')}"
            for s in st.session_state.get("story_history", [])[-2:]
        ])

        if api_key:
            prompt = f"""
You are the creative storytelling AI for "Dastaan Canvas", combining the warmth and magical wonder of Toy Story with high-aesthetic modern studio visual narratives.

Story Context:
- Character (LOCKED): {character['name']} ({character['description']})
- Setting (LOCKED): {setting['name']} ({setting['description']})
- Scene Number: {scene_idx}
- Previous context: {past_scenes_summary if past_scenes_summary else "Beginning of the quest."}
- User Action/Branch Chosen: "{chosen_branch if chosen_branch else user_input}"
- User Detected Emotion: {emotion_name.upper()} (Valence: {valence:.2f}, Arousal: {arousal:.2f})
- Required Story Tone: {story_tone}
- Language: {language} (Provide story narrative and choices strictly in {language})

Task:
1. Write Scene {scene_idx} narrative (approx 80-120 words). If emotion is sad/melancholic, make pacing gentle, reflective and intimate. If joyful, make it wondrous, lively and energetic.
2. Formulate an AI Image Prompt for this scene that maintains strict visual continuity:
   - Must include character: "{character['name']}, {character['description']}"
   - Must include setting: "{setting['name']}, {setting['description']}"
   - Art style: "{brush_texture}, {camera_style}, color palette dominated by {palette_str}, 3D Pixar & Toy Story cinematic lighting, octane render, 8k masterpiece"
3. Provide 2 to 3 branching choices for the user to steer the story next.
4. Give a 1-sentence artistic rationale ("why_this_image") explaining how the scene mirrors the user's emotion.

Output strictly valid JSON matching this schema:
{{
  "scene_number": {scene_idx},
  "title": "A captivating title in {language}",
  "narrative": "The story text in {language}",
  "short_summary": "1 sentence plot summary",
  "image_prompt": "Highly detailed visual prompt in English for diffusion models",
  "branching_choices": [
    "Choice A text in {language}",
    "Choice B text in {language}",
    "Choice C text in {language}"
  ],
  "why_this_image": "Explanation of how emotion and visual cues converge"
}}
"""
            try:
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.7, "response_mime_type": "application/json"}
                }
                for m in CANDIDATE_MODELS:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
                        res = requests.post(url, headers=headers, json=payload, timeout=14)
                        if res.status_code == 200:
                            raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
                            data = json.loads(cleaned)
                            data["emotion"] = emotion_name
                            data["valence"] = valence
                            data["arousal"] = arousal
                            data["palette"] = cfg.get("palette", [])
                            return data
                    except Exception:
                        continue
            except Exception:
                pass

        # Robust High-Quality Procedural Story Generator Fallback
        return cls._procedural_story_fallback(
            scene_idx=scene_idx,
            character=character,
            setting=setting,
            emotion_name=emotion_name,
            valence=valence,
            arousal=arousal,
            cfg=cfg,
            chosen_branch=chosen_branch or user_input,
            language=language
        )

    @classmethod
    def _procedural_story_fallback(cls, scene_idx, character, setting, emotion_name, valence, arousal, cfg, chosen_branch, language):
        """Generates rich, contextual story scenes when offline."""
        palette_str = ", ".join(cfg.get("palette", []))
        brush_texture = cfg.get("brush_texture", "luminous soft painterly rendering")
        camera_style = cfg.get("camera_style", "cinematic depth of field")

        if language == "Hindi":
            if emotion_name in ["sadness", "nostalgia"]:
                title = f"दृश्य {scene_idx}: भीगी यादों का झरोखा"
                narrative = (
                    f"खामोशी की धीमी चादर ओढ़े, {character['name']} ने {setting['name']} की पुरानी खिड़की से बाहर देखा। "
                    f"हवा में बीते पलों की मीठी महक थी। दिल में हल्का सा सूनापन था, मानो कोई भूली हुई धुन दिल की गहराइयों में बज रही हो। "
                    f"हर छोटा खिलौना और पुरानी तस्वीर कह रही थी कि कुछ कहानियां कभी खत्म नहीं होतीं, बस एक नए मोड़ का इंतज़ार करती हैं।"
                )
                choices = [
                    "खिड़की पर बैठकर बारिश की बूंदों को सुनना",
                    "पुरानी संदूक खोलकर भूली यादों को टटोलना",
                    "एक नन्हा दिया जलाकर रौशनी की तलाश करना"
                ]
            else:
                title = f"दृश्य {scene_idx}: सुनहरी उम्मीद का आगाज़"
                narrative = (
                    f"अचानक एक चमचमाती किरण कमरे में उतरी! {character['name']} के पंखों में नई जान आ गई। "
                    f"{setting['name']} के हर कोने में रंगीन तितलियां नाच उठीं। "
                    f"उमंग और उत्साह से भरी एक नई यात्रा सामने मुस्कुरा रही थी। आगे कदम बढ़ाते ही हर धड़कन में जादू भर गया!"
                )
                choices = [
                    "उड़ान भरकर उस सुनहरी रौशनी का पीछा करना",
                    "कमरे में छिपे जादूई नक्शे को खोलना",
                    "अपने दोस्तों को नई सुबह का बुलावा देना"
                ]
        elif language == "Hinglish":
            if emotion_name in ["sadness", "nostalgia"]:
                title = f"Scene {scene_idx}: Ek Thahri Hui Shaam"
                narrative = (
                    f"{setting['name']} mein ek shaant si khamoshi phaili hui thi. {character['name']} ne dheere se apni aankhein uthayin. "
                    f"Purani yaadein amber lights ki tarah chamak rahi thi. Dil thoda udaas tha, lekin ek pyari si umeed bhi thi "
                    f"ki har andheri raat ke baad ek warm aur luminous subah zaroor aati hai."
                )
                choices = [
                    "Quiet corner mein baith kar purani dhun sunna",
                    "Ek purana golden lock kholne ki koshish karna",
                    "Ek cup chai ki khushbu ke saath aage badhna"
                ]
            else:
                title = f"Scene {scene_idx}: Magic aur Adventure Ka Spark"
                narrative = (
                    f"Ek dum se vibrant colors hawa mein explode hue! {character['name']} excitement ke saath jump kar gaya. "
                    f"{setting['name']} ab ek magical wonder-land ban chuka tha. "
                    f"Aisa laga jaise canvas khud bol raha ho—har pixel mein energy aur smiles thi!"
                )
                choices = [
                    "Cosmic portal ke through fly karna",
                    "Secret magical gadget ko activate karna",
                    "Toy kingdom ke sabhi doston ko cheer karna"
                ]
        else:
            # English
            if emotion_name in ["sadness", "fear"]:
                title = f"Scene {scene_idx}: The Whispering Shadows of Solitude"
                narrative = (
                    f"A gentle hush settled over {setting['name']}. {character['name']} paused beneath the indigo twilight, "
                    f"listening to the rhythmic tap of raindrops against the glass. Even amidst the solemn blue silence, "
                    f"a tender spark of resilience flickered—a quiet reminder that every deep shadow is sculpted by a hidden light."
                )
                choices = [
                    "Gently inspect the glowing sapphire compass in the dark",
                    "Follow the soft melody humming beneath the floorboards",
                    "Light an amber tallow candle to illuminate the forgotten shelf"
                ]
            elif emotion_name in ["anger"]:
                title = f"Scene {scene_idx}: The Forge of Fiery Resolve"
                narrative = (
                    f"Crimson sparks danced across {setting['name']}. {character['name']} braced against the roaring wind, "
                    f"gears whirling with unyielding determination. The storm was fierce, yet in that turbulence lay the power "
                    f"to break through any ancient cage and forge an unshakeable path forward."
                )
                choices = [
                    "Charge boldly through the swirling embers",
                    "Reroute the mechanical dynamo to stabilize the engine",
                    "Confront the storm head-on with a resonant call"
                ]
            else:
                title = f"Scene {scene_idx}: The Golden Bloom of Wonder"
                narrative = (
                    f"Golden morning light burst across {setting['name']} like warm liquid amber! {character['name']} leaped forward, "
                    f"its mechanical wings catching the luminous breeze. Floating stardust twirled in the air, transforming the space into "
                    f"an enchanted playground where every forgotten treasure pulsed with joyful life."
                )
                choices = [
                    "Soar upward into the golden sunbeam portal",
                    "Unfurl the celestial star map glowing on the floor",
                    "Awaken the sleeping musical carousel beside the window"
                ]

        image_prompt = (
            f"Cinematic 3D animation still in the beloved style of Toy Story and modern Pixar masterworks. "
            f"Central subject: {character['name']}, {character['description']}. "
            f"Environment: {setting['name']}, {setting['description']}. "
            f"Lighting & Mood: {emotion_name.upper()} atmosphere, {camera_style}, {brush_texture}, "
            f"harmonious color palette featuring {palette_str}, rich tactile textures, subsurface scattering, 8k render."
        )

        return {
            "scene_number": scene_idx,
            "title": title,
            "narrative": narrative,
            "short_summary": f"{character['name']} encounters a {emotion_name} transition within {setting['name']}.",
            "image_prompt": image_prompt,
            "branching_choices": choices,
            "why_this_image": f"Visual dynamics mirror '{emotion_name}' (valence: {valence:.2f}, arousal: {arousal:.2f}) with {camera_style} and {palette_str} color balance.",
            "emotion": emotion_name,
            "valence": valence,
            "arousal": arousal,
            "palette": cfg.get("palette", [])
        }
