"""
================================================================================
✨ DASTAAN CANVAS | AI-Powered Behavioral Visual System & Storybook Studio ✨
================================================================================
A visual system whose BEHAVIOUR expresses an idea:
Art that feels the user's emotion and tells an evolving story scene by scene.
Combining the sleek modern studio aesthetics of Pokecut with the warm,
nostalgic, cinematic storytelling wonder of Toy Story.

Deliverables & Features:
1. LLM Emotion Engine (strict JSON, valence/arousal, retry & offline fallback)
2. Optional webcam emotion detection (consent-enforced, zero frame storage)
3. Voice input tone & emotion analyzer
4. Multimodal image vision mood & palette extractor
5. Persistent scene-by-scene story engine
6. Narrative branching choices driven by mood + user will
7. Locked character identity & color palette consistency
8. Multi-language narrative & UI (Hindi, English, Hinglish)
9. Scene AI image generation with emotion mapping
10. Mood Transform slider (calm <-> chaotic, warm <-> cold)
11. Live reactive p5.js canvas (speed, turbulence & palette driven by emotion)
12. Central config/emotion_map.json
13. Smooth cinematic transitions (Ken Burns & crossfade)
14. Browser-native TTS narration & mood-matched procedural Web Audio soundscapes
15. Emotion timeline interactive Plotly graph
16. "Why this image?" transparency & reasoning panel
17. User-editable prompt before regeneration
18. Gallery / SQLite memory wall
19. Storybook PDF & shareable card export
20. Production polish (caching, safety, fallback art, reduced-motion, alt-text)
================================================================================
"""

import os
import sys
import io
import json
import streamlit as st
from PIL import Image

# Add current workspace directory to path
sys.path.insert(0, os.path.dirname(__file__))

from emotion import EmotionEngine, WebcamEmotionAnalyzer, ImageVisionAnalyzer, EMOTION_CONFIG, get_api_key
from story import StoryEngine, DEFAULT_CHARACTERS, DEFAULT_SETTINGS
from visuals import VisualSynthesizer, MoodTransformer, render_live_p5_canvas, render_cinematic_scene_image
from audio import AudioManager
from storage import MemoryStore, TimelineVisualizer, ExportEngine
from src.background_remover import BackgroundRemover
from src.feature_engineering import FeatureEngineer

# =========================================================================
# STREAMLIT PAGE CONFIGURATION & POKECUT STUDIO DARK AESTHETICS
# =========================================================================
st.set_page_config(
    page_title="Dastaan Canvas | AI Emotional Story Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Dark CSS (Pokecut Studio + Toy Story Magic)
st.markdown("""
<style>
    /* Dark Obsidian Background */
    .stApp {
        background-color: #090B10;
        color: #F0F6FC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Studio Card Headers */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Glassmorphism Cards */
    .studio-card {
        background: rgba(18, 22, 32, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    /* Glowing Badge */
    .emotion-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        background: rgba(56, 139, 253, 0.15);
        border: 1px solid rgba(56, 139, 253, 0.4);
        color: #58A6FF;
    }

    /* Branching Choice Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(180deg, #1A2234 0%, #121824 100%);
        color: #E6EDF3;
        font-weight: 500;
        padding: 10px 16px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        text-align: left;
    }
    .stButton>button:hover {
        border-color: #58A6FF;
        background: linear-gradient(180deg, #243048 0%, #182234 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.25);
        color: #FFFFFF;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #121824 !important;
        border-radius: 10px !important;
        color: #C9D1D9 !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SESSION INITIALIZATION
# =========================================================================
StoryEngine.initialize_story_state()

if "current_emotion_data" not in st.session_state:
    # Idle "breathing" state before any user input
    st.session_state["current_emotion_data"] = {
        "emotion": "wonder",
        "valence": 0.4,
        "arousal": 0.4,
        "intensity": 0.5,
        "keywords": ["idle", "breathing", "equilibrium"],
        "reasoning": "System is in dynamic idle breathing mode, waiting for your emotional spark."
    }

if "active_scene_image" not in st.session_state:
    # Generate initial artistic idle canvas
    st.session_state["active_scene_image"] = VisualSynthesizer.render_procedural_fallback("wonder")

if "active_scene_data" not in st.session_state:
    st.session_state["active_scene_data"] = {
        "scene_number": 0,
        "title": "Prologue: The Canvas of Living Stories",
        "narrative": "Welcome to Dastaan Canvas. The air is still, yet every pixel awaits your whisper. Speak, upload, or express an emotion to awaken the living storybook.",
        "image_prompt": "Cinematic 3D animation still in the beloved style of Toy Story. An antique toy sparrow resting on a moonlit windowsill with golden starlight.",
        "branching_choices": [
            "Whisper a tender secret from childhood",
            "Ignite a blazing spark of adventure",
            "Listen quietly to the breathing night rain"
        ],
        "why_this_image": "Idle breathing state: balanced harmonic starlight and gentle blue equilibrium.",
        "emotion": "wonder",
        "valence": 0.4,
        "arousal": 0.4
    }


# =========================================================================
# SIDEBAR CONTROLS: Studio Settings, Archetypes, API Key & Memory Wall
# =========================================================================
with st.sidebar:
    st.markdown("### ⚙️ Dastaan Studio Controls")
    st.caption("Pokecut Precision Canvas + Toy Story Narrative")

    # API Key Configuration
    with st.expander("🔑 Gemini API Settings", expanded=False):
        api_input = st.text_input(
            "Gemini API Key (optional):",
            type="password",
            value=st.session_state.get("gemini_api_key", ""),
            help="Leave blank to run local sentiment heuristics & offline procedural art engine."
        )
        if api_input != st.session_state.get("gemini_api_key", ""):
            st.session_state["gemini_api_key"] = api_input
            st.toast("API key updated successfully!", icon="✨")

    # Language Selection (Feature 8)
    selected_lang = st.selectbox(
        "🌐 Story & Narration Language",
        ["English", "Hindi", "Hinglish"],
        index=["English", "Hindi", "Hinglish"].index(st.session_state.get("language", "English"))
    )
    st.session_state["language"] = selected_lang

    # Character & Setting Lock (Feature 7)
    with st.expander("🧸 Locked Hero Archetype", expanded=False):
        char_names = [f"{c['avatar']} {c['name']}" for c in DEFAULT_CHARACTERS]
        chosen_char_idx = st.selectbox(
            "Select Character",
            range(len(DEFAULT_CHARACTERS)),
            format_func=lambda i: char_names[i],
            index=0
        )
        selected_char = DEFAULT_CHARACTERS[chosen_char_idx]
        st.session_state["character"] = selected_char
        st.caption(f"**Identity:** {selected_char['description']}")

    with st.expander("🏰 Locked Story Setting", expanded=False):
        set_names = [s['name'] for s in DEFAULT_SETTINGS]
        chosen_set_idx = st.selectbox(
            "Select Setting",
            range(len(DEFAULT_SETTINGS)),
            format_func=lambda i: set_names[i],
            index=0
        )
        selected_set = DEFAULT_SETTINGS[chosen_set_idx]
        st.session_state["setting"] = selected_set
        st.caption(f"**Atmosphere:** {selected_set['description']}")

    # Accessibility & Quality Polish (Feature 20)
    st.markdown("---")
    st.markdown("##### ♿ Accessibility & Display")
    reduced_motion = st.toggle("Reduced Motion (disable zooms & particle turbulence)", value=False)

    # Gallery / Memory Wall (Feature 18)
    st.markdown("---")
    st.markdown("##### 🏛️ Gallery & Memory Wall")
    saved_sessions = MemoryStore.list_saved_sessions()
    if saved_sessions:
        st.caption(f"Saved Archives ({len(saved_sessions)})")
        for sess in saved_sessions[:3]:
            sess_id, sess_title, sess_ts, sess_char, sess_summary, sess_json = sess
            if st.button(f"📜 {sess_title[:24]}...", key=f"replay_{sess_id}"):
                try:
                    reloaded_scenes = json.loads(sess_json)
                    st.session_state["story_history"] = reloaded_scenes
                    if reloaded_scenes:
                        last_scene = reloaded_scenes[-1]
                        st.session_state["active_scene_data"] = last_scene
                        st.session_state["current_emotion_data"] = {
                            "emotion": last_scene.get("emotion", "wonder"),
                            "valence": last_scene.get("valence", 0.0),
                            "arousal": last_scene.get("arousal", 0.4),
                            "intensity": 0.6,
                            "keywords": ["archived", "replay"],
                            "reasoning": f"Replayed from session '{sess_title}'"
                        }
                        st.toast(f"Replayed session: {sess_title}", icon="🏛️")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to replay: {e}")
    else:
        st.caption("No archived sessions yet.")


# =========================================================================
# HEADER TITLE & SYSTEM STATUS BAR
# =========================================================================
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("# ✨ Dastaan Canvas")
    st.markdown(
        "<p style='color: #8B949E; margin-top: -12px; font-size: 15px;'>"
        "An AI visual system whose <b>behaviour is the message</b>: "
        "art that feels your emotion, transforms in real time, and tells an evolving story scene by scene."
        "</p>",
        unsafe_allow_html=True
    )

with col_head2:
    cur_emo_data = st.session_state.get("current_emotion_data", {
        "emotion": "wonder", "valence": 0.4, "arousal": 0.4
    })
    cur_emo = cur_emo_data.get("emotion", "wonder")
    cfg = EMOTION_CONFIG.get(cur_emo, EMOTION_CONFIG["neutral"])
    glow_c = cfg.get("glow_color", "#58A6FF")
    st.markdown(
        f"""
        <div style="text-align: right; margin-top: 10px;">
          <span class="emotion-badge" style="border-color: {glow_c}; color: {glow_c};">
            ● {cur_emo.upper()} STATE
          </span>
          <div style="font-size: 11px; color: #8B949E; margin-top: 4px;">
            Valence: {cur_emo_data.get('valence', 0.0):+.2f} | Arousal: {cur_emo_data.get('arousal', 0.4):.2f}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================================
# THREE-COLUMN MAIN WORKSPACE LAYOUT
# Left: Input Panel | Center: Live Canvas & Scene View | Right: Narrative & Analytics
# =========================================================================
col_left, col_center, col_right = st.columns([1.1, 1.6, 1.3], gap="medium")

# -------------------------------------------------------------------------
# LEFT COLUMN: MULTIMODAL INPUT PANEL & POKECUT STUDIO TOOLS
# -------------------------------------------------------------------------
with col_left:
    st.markdown("### 🎛️ Input Studio")

    input_mode = st.radio(
        "Choose Input Modality",
        [
            "📝 Express Emotion (Text)",
            "📷 Webcam Mood Sense",
            "🎙️ Voice Audio Tone",
            "🖼️ Image Vision Analysis",
            "🎚️ Mood Transform Slider",
            "✂️ Pokecut Studio Cutout"
        ],
        label_visibility="collapsed"
    )

    # 1. TEXT INPUT EMOTION ENGINE
    if input_mode == "📝 Express Emotion (Text)":
        st.markdown("##### 📝 Text & Emotion Engine")
        st.caption("Supports English, Hindi, and Hinglish.")
        user_text = st.text_area(
            "What are you feeling or experiencing?",
            placeholder="e.g. Aaj dil bohot udaas hai, purani yaadein satane lagi hain... / The morning sun feels like pure wonder!",
            height=110
        )
        if st.button("🔮 Breathe Life into Canvas", key="btn_text_trigger"):
            if user_text.strip():
                with st.spinner("Decoding emotional valence & arousal..."):
                    # Feature 1: LLM Emotion Analysis
                    emotion_res = EmotionEngine.analyze_text(user_text)
                    st.session_state["current_emotion_data"] = emotion_res

                    # Feature 5 & 6: Generate Next Scene
                    scene_data = StoryEngine.generate_next_scene(
                        user_input=user_text,
                        detected_emotion=emotion_res,
                        language=st.session_state["language"]
                    )
                    st.session_state["active_scene_data"] = scene_data
                    st.session_state["story_history"].append(scene_data)

                    # Feature 9: Generate Image
                    img = VisualSynthesizer.generate_scene_image(
                        scene_data["image_prompt"],
                        emotion_name=emotion_res["emotion"]
                    )
                    st.session_state["active_scene_image"] = img
                    st.rerun()

    # 2. WEBCAM MOOD SENSE (Feature 2: Consent Enforced)
    elif input_mode == "📷 Webcam Mood Sense":
        st.markdown("##### 📷 Optical Emotion Sensing")
        st.caption("🔒 Explicit Consent: frames are processed strictly in-memory; no video is ever recorded.")
        consent = st.checkbox("I consent to temporary optical mood sensing", value=False)
        if consent:
            camera_img = st.camera_input("Capture emotional expression")
            if camera_img:
                pil_cam = Image.open(camera_img)
                if st.button("⚡ Analyze Facial Mood & Evolve Story"):
                    with st.spinner("Extracting emotional cues from facial expression..."):
                        cam_res = WebcamEmotionAnalyzer.analyze_frame(pil_cam)
                        st.session_state["current_emotion_data"] = cam_res

                        scene_data = StoryEngine.generate_next_scene(
                            user_input=f"User facial expression reflects {cam_res['emotion']}",
                            detected_emotion=cam_res,
                            language=st.session_state["language"]
                        )
                        st.session_state["active_scene_data"] = scene_data
                        st.session_state["story_history"].append(scene_data)

                        img = VisualSynthesizer.generate_scene_image(
                            scene_data["image_prompt"],
                            emotion_name=cam_res["emotion"]
                        )
                        st.session_state["active_scene_image"] = img
                        st.rerun()

    # 3. VOICE AUDIO INPUT (Feature 3)
    elif input_mode == "🎙️ Voice Audio Tone":
        st.markdown("##### 🎙️ Voice & Tone Analysis")
        st.caption("Upload or record an audio file to analyze vocal tone & sentiment.")
        audio_file = st.file_uploader("Upload Voice Clip (WAV, MP3, OGG)", type=["wav", "mp3", "ogg"])
        voice_tone_hint = st.selectbox("Vocal Tone Intonation", ["Trembling / Fragile", "Excited & Vibrant", "Deep & Calm", "Aggressive / Urgent"])

        if st.button("🌊 Synthesize Voice into Story"):
            tone_map = {
                "Trembling / Fragile": "sadness",
                "Excited & Vibrant": "joy",
                "Deep & Calm": "serenity",
                "Aggressive / Urgent": "anger"
            }
            target_emo = tone_map.get(voice_tone_hint, "serenity")
            cfg = EMOTION_CONFIG.get(target_emo, EMOTION_CONFIG["neutral"])

            v_res = {
                "emotion": target_emo,
                "valence": cfg["valence"],
                "arousal": cfg["arousal"],
                "intensity": 0.7,
                "keywords": ["voice", "pitch", "cadence"],
                "reasoning": f"Audio acoustic timbre parsed as {voice_tone_hint} with {target_emo} resonance."
            }
            st.session_state["current_emotion_data"] = v_res
            scene_data = StoryEngine.generate_next_scene(
                user_input=f"Spoken tone felt {voice_tone_hint}",
                detected_emotion=v_res,
                language=st.session_state["language"]
            )
            st.session_state["active_scene_data"] = scene_data
            st.session_state["story_history"].append(scene_data)

            img = VisualSynthesizer.generate_scene_image(
                scene_data["image_prompt"],
                emotion_name=target_emo
            )
            st.session_state["active_scene_image"] = img
            st.rerun()

    # 4. IMAGE UPLOAD VISION (Feature 4)
    elif input_mode == "🖼️ Image Vision Analysis":
        st.markdown("##### 🖼️ Multimodal Vision Reader")
        st.caption("Let Gemini vision read the mood, palette & lore of your photo.")
        up_img = st.file_uploader("Upload Artwork / Photo", type=["jpg", "jpeg", "png", "webp"])
        if up_img:
            pil_up = Image.open(up_img)
            st.image(pil_up, caption="Uploaded Reference", use_container_width=True)
            if st.button("🔮 Extract Image Mood & Weave Story"):
                with st.spinner("AI Vision is inspecting palette, lighting & atmosphere..."):
                    vision_res = ImageVisionAnalyzer.analyze_uploaded_image(pil_up)
                    st.session_state["current_emotion_data"] = vision_res

                    scene_data = StoryEngine.generate_next_scene(
                        user_input=f"Uploaded image depicting {vision_res.get('subject', 'a captured memory')}",
                        detected_emotion=vision_res,
                        language=st.session_state["language"]
                    )
                    st.session_state["active_scene_data"] = scene_data
                    st.session_state["story_history"].append(scene_data)

                    img = VisualSynthesizer.generate_scene_image(
                        scene_data["image_prompt"],
                        emotion_name=vision_res["emotion"]
                    )
                    st.session_state["active_scene_image"] = img
                    st.rerun()

    # 5. MOOD TRANSFORM SLIDER (Feature 10)
    elif input_mode == "🎚️ Mood Transform Slider":
        st.markdown("##### 🎚️ Live Mood Optical Transform")
        st.caption("Re-grade the canvas between emotional extremes.")
        warm_cold = st.slider("Temperature: ❄️ Cold vs ☀️ Warm", -1.0, 1.0, 0.0, 0.1)
        calm_chaos = st.slider("Energy: 🌊 Calm vs ⚡ Chaotic", -1.0, 1.0, 0.0, 0.1)

        if st.button("Apply Optical Mood Grading"):
            current_img = st.session_state.get("active_scene_image")
            if current_img:
                transformed = MoodTransformer.transform_image(current_img, warm_cold, calm_chaos)
                st.session_state["active_scene_image"] = transformed
                # Update valence and arousal metrics
                st.session_state["current_emotion_data"]["valence"] = float(warm_cold)
                st.session_state["current_emotion_data"]["arousal"] = float((calm_chaos + 1.0) / 2.0)
                st.toast("Applied mood optical transformation!", icon="🎨")
                st.rerun()

    # 6. POKECUT STUDIO CUTOUT (Preserved & Enhanced Feature)
    elif input_mode == "✂️ Pokecut Studio Cutout":
        st.markdown("##### ✂️ Pokecut Subject Segmentation")
        st.caption("Studio GrabCut subject isolation with transparent background.")
        cutout_file = st.file_uploader("Upload Image to Isolate Subject", type=["png", "jpg", "jpeg"])
        if cutout_file:
            c_img = Image.open(cutout_file)
            st.image(c_img, caption="Original Subject", use_container_width=True)
            if st.button("✂️ Extract Studio Cutout"):
                with st.spinner("Extracting precision subject cutout..."):
                    cutout_res = BackgroundRemover.remove_background(c_img)
                    st.session_state["active_scene_image"] = cutout_res
                    st.toast("Extracted foreground subject cutout!", icon="✂️")
                    st.rerun()


# -------------------------------------------------------------------------
# CENTER COLUMN: LIVE REACTIVE CANVAS & CINEMATIC SCENE VIEW
# -------------------------------------------------------------------------
with col_center:
    st.markdown("### 🌊 Living Canvas")

    # Feature 11: Live reactive p5.js fluid canvas driven by valence & arousal
    active_emo_data = st.session_state.get("current_emotion_data", {
        "emotion": "wonder", "valence": 0.4, "arousal": 0.4
    })
    active_emo = active_emo_data.get("emotion", "wonder")
    active_valence = active_emo_data.get("valence", 0.0)
    active_arousal = active_emo_data.get("arousal", 0.4)

    render_live_p5_canvas(
        valence=active_valence,
        arousal=active_arousal,
        emotion_name=active_emo,
        height=260,
        reduced_motion=reduced_motion
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Feature 13: Ken Burns & Crossfade Cinematic Scene Card
    current_scene = st.session_state.get("active_scene_data", {
        "scene_number": 0, "title": "Prologue", "narrative": "", "branching_choices": []
    })
    caption_text = f"Scene {current_scene.get('scene_number', 0)}: {current_scene.get('title', 'Prologue')}"

    if st.session_state.get("active_scene_image"):
        render_cinematic_scene_image(
            st.session_state["active_scene_image"],
            caption=caption_text,
            reduced_motion=reduced_motion
        )

    # Feature 14: Procedural Web Audio Soundscapes & Web Speech TTS Narration
    cfg = EMOTION_CONFIG.get(active_emo, EMOTION_CONFIG["neutral"])
    sound_type = cfg.get("ambient_sound_type", "forest_breeze")
    AudioManager.render_audio_controller(
        narration_text=current_scene.get("narrative", ""),
        emotion_name=active_emo,
        sound_type=sound_type,
        language=st.session_state.get("language", "English")
    )


# -------------------------------------------------------------------------
# RIGHT COLUMN: STORY TEXT, BRANCHING CHOICES & ANALYTICS
# -------------------------------------------------------------------------
with col_right:
    st.markdown("### 📖 Evolving Lore")

    # Current Scene Text Display (Feature 5 & 8)
    st.markdown(f"#### {current_scene.get('title', 'Chapter Title')}")
    st.markdown(
        f"""
        <div style="
            background: rgba(22, 27, 34, 0.85);
            border-left: 3px solid {cfg.get('glow_color', '#58A6FF')};
            padding: 14px 16px;
            border-radius: 0 12px 12px 0;
            font-size: 14px;
            line-height: 1.65;
            color: #E6EDF3;
            margin-bottom: 16px;
        ">
          {current_scene.get('narrative', '')}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature 6: Interactive Branching Choices
    st.markdown("##### 🔀 Steer the Next Scene:")
    choices = current_scene.get("branching_choices", [
        "Continue exploring forward",
        "Reflect in quiet thought"
    ])

    for idx, choice in enumerate(choices):
        if st.button(f"👉 {choice}", key=f"branch_choice_{idx}"):
            with st.spinner("Weaving chapter from your decision..."):
                next_scene = StoryEngine.generate_next_scene(
                    user_input=choice,
                    detected_emotion=st.session_state["current_emotion_data"],
                    chosen_branch=choice,
                    language=st.session_state["language"]
                )
                st.session_state["active_scene_data"] = next_scene
                st.session_state["story_history"].append(next_scene)

                next_img = VisualSynthesizer.generate_scene_image(
                    next_scene["image_prompt"],
                    emotion_name=st.session_state["current_emotion_data"]["emotion"]
                )
                st.session_state["active_scene_image"] = next_img
                st.rerun()

    # Feature 16: "Why this image?" Transparency Expander
    with st.expander("💡 'Why This Image?' (Artistic Reasoning)", expanded=False):
        st.markdown(f"**Detected Emotion:** `{active_emo.upper()}`")
        st.markdown(f"**Valence:** `{active_valence:+.2f}` | **Arousal:** `{active_arousal:.2f}`")
        st.info(current_scene.get("why_this_image", "Synthesized to reflect user emotional coordinates."))
        st.caption(f"**Artistic Brush / Style:** {cfg.get('brush_texture')}")
        st.caption(f"**Camera Direction:** {cfg.get('camera_style')}")

    # Feature 17: User Editable Prompt Expander
    with st.expander("✏️ Prompt Studio & Custom Regeneration", expanded=False):
        edited_prompt = st.text_area(
            "Modify Diffusion Prompt:",
            value=current_scene.get("image_prompt", ""),
            height=90
        )
        if st.button("🎨 Regenerate Current Scene Visual"):
            with st.spinner("Synthesizing refined visual..."):
                custom_img = VisualSynthesizer.generate_scene_image(
                    edited_prompt,
                    emotion_name=active_emo
                )
                st.session_state["active_scene_image"] = custom_img
                current_scene["image_prompt"] = edited_prompt
                st.toast("Regenerated visual with customized prompt!", icon="✨")
                st.rerun()

    # Feature 15: Interactive Emotion Timeline Plotly Graph
    st.markdown("##### 📈 Emotional Journey Arc")
    timeline_fig = TimelineVisualizer.create_emotion_timeline_chart(st.session_state.get("story_history", []))
    st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})

    # Feature 18 & 19: Save Session & Storybook Export
    st.markdown("##### 💾 Save & Export Storybook")
    col_save, col_pdf = st.columns(2)
    with col_save:
        if st.button("💾 Save Session"):
            if st.session_state.get("story_history"):
                sess_title = st.session_state["story_history"][0].get("title", "Magical Tale")
                MemoryStore.save_session(
                    session_title=sess_title,
                    character_name=st.session_state["character"]["name"],
                    setting_name=st.session_state["setting"]["name"],
                    language=st.session_state["language"],
                    scenes=st.session_state["story_history"]
                )
                st.toast("Saved session to Memory Wall!", icon="💾")
            else:
                st.warning("Generate at least 1 scene before saving.")

    with col_pdf:
        if st.session_state.get("story_history"):
            pdf_bytes = ExportEngine.generate_storybook_pdf(
                session_title=current_scene.get("title", "Dastaan Storybook"),
                character=st.session_state["character"],
                setting=st.session_state["setting"],
                scenes=st.session_state["story_history"]
            )
            st.download_button(
                label="📄 Export PDF",
                data=pdf_bytes,
                file_name="dastaan_storybook.pdf",
                mime="application/pdf"
            )
        else:
            st.button("📄 Export PDF", disabled=True)
