"""
================================================================================
✨ DASTAAN CANVAS | Advanced AI Visual System & Story Studio ✨
================================================================================
A high-aesthetic, intelligent visual platform uniting Pokecut's studio precision
with the cinematic emotional storytelling wonder of animated cinema.

Proprietary In-House Neural Architecture:
- Cognitive Emotion Engine (Continuous Valence & Arousal Mapping)
- Dynamic Reactive Flow Canvas (Real-time particle physics driven by sentiment)
- Scene-by-Scene Branching Story Engine with Persistent Character Lock
- Studio Subject Cutout & Background Isolation (GrabCut Segmentation)
- Optical Mood & Style Color Grading Pipeline
- Ambient Audio Synthesizer & Natural Speech Narration
- Interactive Emotional Journey Arc & Illustrated Storybook PDF Export
================================================================================
"""

import os
import sys
import io
import json
import streamlit as st
import numpy as np
from PIL import Image

# Workspace imports
sys.path.insert(0, os.path.dirname(__file__))

from emotion import EmotionEngine, WebcamEmotionAnalyzer, ImageVisionAnalyzer, EMOTION_CONFIG
from story import StoryEngine, DEFAULT_CHARACTERS, DEFAULT_SETTINGS
from visuals import VisualSynthesizer, MoodTransformer, render_live_p5_canvas, render_cinematic_scene_image
from audio import AudioManager
from storage import MemoryStore, TimelineVisualizer, ExportEngine
from src.background_remover import BackgroundRemover
from src.feature_engineering import FeatureEngineer
from src.model_engine import ModelEngine

# =========================================================================
# PAGE SETUP & POKECUT STUDIO DARK GLASSMORPHISM
# =========================================================================
st.set_page_config(
    page_title="Dastaan Canvas | AI Creative Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Polished, Non-cluttered Studio CSS
st.markdown("""
<style>
    /* Dark Studio Background */
    .stApp {
        background-color: #0B0E14;
        color: #F0F6FC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Clean Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(18, 24, 38, 0.7);
        padding: 8px 12px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        color: #8B949E;
        font-weight: 600;
        font-size: 14px;
        padding: 0 18px;
        border: none !important;
        background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1F6FEB 0%, #388BFD 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(31, 111, 235, 0.35);
    }

    /* Studio Card */
    .studio-panel {
        background: rgba(16, 21, 32, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    /* Live Emotional Badge */
    .neural-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        background: rgba(88, 166, 255, 0.12);
        border: 1px solid rgba(88, 166, 255, 0.3);
        color: #58A6FF;
    }

    /* Branching Story Action Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(180deg, #1A2333 0%, #121824 100%);
        color: #E6EDF3;
        font-weight: 500;
        padding: 10px 16px;
        transition: all 0.2s ease;
        text-align: left;
    }
    .stButton>button:hover {
        border-color: #58A6FF;
        background: linear-gradient(180deg, #25334D 0%, #182234 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.2);
        color: #FFFFFF;
    }

    /* Story Text Frame */
    .story-card {
        background: rgba(22, 28, 42, 0.85);
        border-left: 4px solid #58A6FF;
        padding: 18px 20px;
        border-radius: 0 14px 14px 0;
        font-size: 15px;
        line-height: 1.7;
        color: #E6EDF3;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SESSION STATE INITIALIZATION
# =========================================================================
StoryEngine.initialize_story_state()

if "current_emotion_data" not in st.session_state:
    st.session_state["current_emotion_data"] = {
        "emotion": "wonder",
        "valence": 0.4,
        "arousal": 0.4,
        "intensity": 0.5,
        "keywords": ["idle", "breathing"],
        "reasoning": "Proprietary Neural Engine in gentle harmonic equilibrium."
    }

if "active_scene_image" not in st.session_state:
    st.session_state["active_scene_image"] = VisualSynthesizer.render_procedural_fallback("wonder")

if "active_scene_data" not in st.session_state:
    st.session_state["active_scene_data"] = {
        "scene_number": 0,
        "title": "Prologue: Awakening the Living Canvas",
        "narrative": "Welcome to Dastaan Canvas. The visual system is currently breathing in serene equilibrium. Express your feeling or upload a memory to weave an evolving cinematic tale.",
        "image_prompt": "Cinematic 3D animation still in beloved animated toy story aesthetic. A clockwork mechanical sparrow resting on a sunlit antique windowsill with floating amber dust motes.",
        "branching_choices": [
            "Whisper a tender forgotten memory",
            "Ignite a blazing spark of adventure",
            "Listen quietly to the breathing twilight"
        ],
        "why_this_image": "Idle equilibrium: balanced golden hour lighting and harmonious gentle particle drift.",
        "emotion": "wonder",
        "valence": 0.4,
        "arousal": 0.4
    }


# =========================================================================
# HEADER & NEURAL PULSE STATUS
# =========================================================================
top_col1, top_col2 = st.columns([3, 1.2])

with top_col1:
    st.markdown("## ✨ Dastaan Canvas")
    st.markdown(
        "<p style='color: #8B949E; margin-top: -14px; font-size: 14px;'>"
        "Next-Generation AI Creative Studio: <b>Art that feels your emotion</b> and unfolds scene by scene."
        "</p>",
        unsafe_allow_html=True
    )

with top_col2:
    cur_emo_data = st.session_state.get("current_emotion_data", {"emotion": "wonder", "valence": 0.4, "arousal": 0.4})
    cur_emo = cur_emo_data.get("emotion", "wonder")
    cfg = EMOTION_CONFIG.get(cur_emo, EMOTION_CONFIG["neutral"])
    glow_col = cfg.get("glow_color", "#58A6FF")
    st.markdown(
        f"""
        <div style="text-align: right; margin-top: 4px;">
          <span class="neural-badge" style="border-color: {glow_col}; color: {glow_col};">
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
# CLEAN SIDEBAR CONTROLS (Hero Archetype, Language, Accessibility)
# =========================================================================
with st.sidebar:
    st.markdown("### ⚙️ Studio Configuration")
    
    # Language Selection (Feature 8)
    selected_lang = st.selectbox(
        "🌐 Narration & Lore Language",
        ["English", "Hindi", "Hinglish"],
        index=["English", "Hindi", "Hinglish"].index(st.session_state.get("language", "English"))
    )
    st.session_state["language"] = selected_lang

    # Hero Archetype Lock (Feature 7)
    st.markdown("##### 🧸 Hero Character Lock")
    char_names = [f"{c['avatar']} {c['name']}" for c in DEFAULT_CHARACTERS]
    chosen_char_idx = st.selectbox(
        "Locked Hero",
        range(len(DEFAULT_CHARACTERS)),
        format_func=lambda i: char_names[i],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state["character"] = DEFAULT_CHARACTERS[chosen_char_idx]

    # Setting Lock
    st.markdown("##### 🏰 Realm Setting")
    set_names = [s['name'] for s in DEFAULT_SETTINGS]
    chosen_set_idx = st.selectbox(
        "Locked Setting",
        range(len(DEFAULT_SETTINGS)),
        format_func=lambda i: set_names[i],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state["setting"] = DEFAULT_SETTINGS[chosen_set_idx]

    # Accessibility & Quality Polish (Feature 20)
    st.markdown("---")
    reduced_motion = st.toggle("Reduced Motion (gentle particles)", value=False)
    st.caption("Engine: Proprietary Neural Vision Core v2.4")


# =========================================================================
# MAIN APP TABS: CLEAR, INTUITIVE & ELEGANT
# =========================================================================
tab_story, tab_cutout, tab_mood, tab_journey = st.tabs([
    "🎨 Living Canvas & Story Studio",
    "✂️ Magic Studio Cutout",
    "🎚️ Mood & Style Transformer",
    "📈 Emotional Journey & Lore PDF"
])


# =========================================================================
# TAB 1: LIVING CANVAS & STORY STUDIO (The Star Multimodal Experience)
# =========================================================================
with tab_story:
    # 1. Clean Input Section
    input_box_col1, input_box_col2 = st.columns([3, 1], gap="medium")

    with input_box_col1:
        story_input = st.text_input(
            "What feeling or vision shall awaken the canvas?",
            placeholder="e.g. Aaj dil thoda udaas hai, purani yaadein aur thandi baarish yaad aa rahi hai... / Golden sunshine bursting through the attic!",
            label_visibility="collapsed"
        )

    with input_box_col2:
        btn_trigger = st.button("✨ Weave Scene & Evolve Art", key="btn_main_trigger")

    # Optional Multimodal Quick Triggers (Expandable only if user wants)
    with st.expander("📷 / 🎙️ / 🖼️ Camera, Voice & Vision Input (Optional)", expanded=False):
        mm_col1, mm_col2 = st.columns(2)
        with mm_col1:
            st.markdown("###### 🖼️ Vision Image Analysis")
            ref_img = st.file_uploader("Upload reference photo to analyze mood", type=["jpg", "png", "jpeg"], key="story_vision_up")
            if ref_img and st.button("🔮 Extract Mood from Image"):
                with st.spinner("Neural Vision analyzing palette & atmosphere..."):
                    pil_ref = Image.open(ref_img)
                    v_res = ImageVisionAnalyzer.analyze_uploaded_image(pil_ref)
                    st.session_state["current_emotion_data"] = v_res
                    next_scene = StoryEngine.generate_next_scene(
                        user_input=f"Uploaded visual depicting {v_res.get('subject', 'a memory')}",
                        detected_emotion=v_res,
                        language=st.session_state["language"]
                    )
                    st.session_state["active_scene_data"] = next_scene
                    st.session_state["story_history"].append(next_scene)
                    st.session_state["active_scene_image"] = VisualSynthesizer.generate_scene_image(
                        next_scene["image_prompt"], emotion_name=v_res["emotion"]
                    )
                    st.rerun()

        with mm_col2:
            st.markdown("###### 📷 Facial Mood Detection")
            cam_consent = st.checkbox("Enable In-Memory Optical Mood Sensing", value=False)
            if cam_consent:
                cam_snap = st.camera_input("Express mood to camera", key="story_cam")
                if cam_snap and st.button("⚡ Read Face & Evolve"):
                    with st.spinner("Analyzing facial cues..."):
                        c_res = WebcamEmotionAnalyzer.analyze_frame(Image.open(cam_snap))
                        st.session_state["current_emotion_data"] = c_res
                        next_scene = StoryEngine.generate_next_scene(
                            user_input="Camera sensed facial emotional state",
                            detected_emotion=c_res,
                            language=st.session_state["language"]
                        )
                        st.session_state["active_scene_data"] = next_scene
                        st.session_state["story_history"].append(next_scene)
                        st.session_state["active_scene_image"] = VisualSynthesizer.generate_scene_image(
                            next_scene["image_prompt"], emotion_name=c_res["emotion"]
                        )
                        st.rerun()

    # Process text input trigger
    if btn_trigger:
        if story_input and story_input.strip():
            with st.spinner("Neural Emotion Core decoding feeling & synthesizing chapter..."):
                emo_res = EmotionEngine.analyze_text(story_input)
                st.session_state["current_emotion_data"] = emo_res

                scene_res = StoryEngine.generate_next_scene(
                    user_input=story_input,
                    detected_emotion=emo_res,
                    language=st.session_state["language"]
                )
                st.session_state["active_scene_data"] = scene_res
                st.session_state["story_history"].append(scene_res)

                art_res = VisualSynthesizer.generate_scene_image(
                    scene_res["image_prompt"],
                    emotion_name=emo_res["emotion"]
                )
                st.session_state["active_scene_image"] = art_res
                st.rerun()
        else:
            st.warning("Please type a feeling or story prompt above before weaving.")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Main Studio Showcase: Split into Visuals (Left) and Lore (Right)
    cur_scene = st.session_state.get("active_scene_data", {})
    active_img = st.session_state.get("active_scene_image")
    active_emo_data = st.session_state.get("current_emotion_data", {"emotion": "wonder", "valence": 0.4, "arousal": 0.4})
    cfg = EMOTION_CONFIG.get(active_emo_data.get("emotion", "wonder"), EMOTION_CONFIG["neutral"])

    show_left, show_right = st.columns([1.35, 1.25], gap="large")

    with show_left:
        st.markdown("##### 🖼️ Synthesized Scene Visual")

        if active_img:
            render_cinematic_scene_image(
                active_img,
                caption=f"Scene {cur_scene.get('scene_number', 0)}: {cur_scene.get('title', 'Prologue')}",
                reduced_motion=reduced_motion
            )

        st.markdown("##### 🌊 Live Emotion-Reactive Canvas")
        # Feature 11: Real-time particle physics driven by Valence & Arousal
        render_live_p5_canvas(
            valence=active_emo_data.get("valence", 0.0),
            arousal=active_emo_data.get("arousal", 0.4),
            emotion_name=active_emo_data.get("emotion", "wonder"),
            height=220,
            reduced_motion=reduced_motion
        )

        # Feature 14: Natural Web Speech TTS & Ambient Soundscape Controller
        AudioManager.render_audio_controller(
            narration_text=cur_scene.get("narrative", ""),
            emotion_name=active_emo_data.get("emotion", "wonder"),
            sound_type=cfg.get("ambient_sound_type", "forest_breeze"),
            language=st.session_state.get("language", "English")
        )

    with show_right:
        st.markdown(f"### {cur_scene.get('title', 'Prologue')}")
        
        # Story Card Display
        st.markdown(
            f"""
            <div class="story-card" style="border-left-color: {cfg.get('glow_color', '#58A6FF')};">
              {cur_scene.get('narrative', '')}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Feature 6: Interactive Branching Choices
        st.markdown("##### 🔀 Steer the Next Scene:")
        choices = cur_scene.get("branching_choices", [
            "Continue exploring forward",
            "Reflect in quiet thought"
        ])

        for idx, choice in enumerate(choices):
            if st.button(f"👉 {choice}", key=f"tab_choice_{idx}"):
                with st.spinner("Weaving the chapter of your choice..."):
                    next_s = StoryEngine.generate_next_scene(
                        user_input=choice,
                        detected_emotion=st.session_state.get("current_emotion_data", {}),
                        chosen_branch=choice,
                        language=st.session_state.get("language", "English")
                    )
                    st.session_state["active_scene_data"] = next_s
                    st.session_state["story_history"].append(next_s)

                    next_art = VisualSynthesizer.generate_scene_image(
                        next_s["image_prompt"],
                        emotion_name=st.session_state.get("current_emotion_data", {}).get("emotion", "wonder")
                    )
                    st.session_state["active_scene_image"] = next_art
                    st.rerun()

        # Clean Inspection Expanders (Features 16 & 17)
        with st.expander("💡 'Why This Image?' (Artistic Reasoning)", expanded=False):
            st.info(cur_scene.get("why_this_image", "Synthesized to mirror detected emotional valence and arousal."))
            st.caption(f"**Brush / Texture:** {cfg.get('brush_texture')}")
            st.caption(f"**Camera Direction:** {cfg.get('camera_style')}")

        with st.expander("✏️ Refine AI Diffusion Prompt", expanded=False):
            edited_p = st.text_area("Custom Scene Prompt:", value=cur_scene.get("image_prompt", ""), height=80)
            if st.button("🎨 Regenerate Visual"):
                with st.spinner("Synthesizing refined visual..."):
                    re_art = VisualSynthesizer.generate_scene_image(edited_p, emotion_name=active_emo_data.get("emotion", "wonder"))
                    st.session_state["active_scene_image"] = re_art
                    cur_scene["image_prompt"] = edited_p
                    st.toast("Updated artwork with custom prompt!", icon="✨")
                    st.rerun()


# =========================================================================
# TAB 2: POKECUT MAGIC STUDIO CUTOUT (Preserved & Enhanced Initial Feature)
# =========================================================================
with tab_cutout:
    st.markdown("#### ✂️ Pokecut Studio Cutout & Subject Isolation")
    st.caption("Advanced foreground segmentation with transparent alpha extraction.")

    cut_file = st.file_uploader("Upload Image to Extract Subject (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"], key="cutout_uploader")

    if cut_file:
        raw_cut_img = Image.open(cut_file)
        
        cut_col1, cut_col2 = st.columns(2, gap="large")
        with cut_col1:
            st.markdown("##### Original Subject")
            st.image(raw_cut_img, use_column_width=True)

        with cut_col2:
            st.markdown("##### Isolated Studio Cutout")
            with st.spinner("Extracting foreground subject with GrabCut precision..."):
                segmented_cutout = BackgroundRemover.remove_background(raw_cut_img)
                st.image(segmented_cutout, use_column_width=True)

                # Download PNG button
                buf = io.BytesIO()
                segmented_cutout.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Cutout (Transparent PNG)",
                    data=buf.getvalue(),
                    file_name="pokecut_studio_cutout.png",
                    mime="image/png"
                )

        # Feature Engineering Metrics (from user's original codebase)
        st.markdown("---")
        st.markdown("##### 📊 Visual Spectrum Metrics")
        features = FeatureEngineer.extract_visual_features(raw_cut_img)
        m1, m2, m3 = st.columns(3)
        m1.metric("Brightness Index", f"{features['brightness']:.1f}")
        m2.metric("Chromatic Saturation", f"{features['saturation']:.1f}")
        m3.metric("Edge Density Clarity", f"{features['edge_density']:.1f}%")
    else:
        st.info("👆 Upload any photo above to instantly isolate the foreground subject with Pokecut studio precision.")


# =========================================================================
# TAB 3: MOOD & STYLE TRANSFORMER (Optical Filters & Temperature Shift)
# =========================================================================
with tab_mood:
    st.markdown("#### 🎚️ Mood & Optical Style Transformer")
    st.caption("Apply real-time cinematic color grading, emotional temperature, and geometric filters.")

    mood_img_source = st.session_state.get("active_scene_image")
    
    t_col1, t_col2 = st.columns([1, 1.4], gap="large")

    with t_col1:
        st.markdown("##### Studio Filter Controls")
        geom_filter = st.selectbox(
            "Geometric Filter Effect",
            ["None", "Grayscale Conversion", "Sharpen Details", "Edge Highlight", "Invert Colors", "Gaussian Blur"]
        )

        warm_val = st.slider("Temperature: ❄️ Cold / Blue vs ☀️ Warm / Amber", -1.0, 1.0, 0.0, 0.1)
        energy_val = st.slider("Dynamics: 🌊 Soft & Serene vs ⚡ Sharp & Chaotic", -1.0, 1.0, 0.0, 0.1)

        apply_btn = st.button("🎨 Render Graded Masterpiece")

    with t_col2:
        st.markdown("##### Graded Canvas Preview")
        if mood_img_source:
            preview_img = mood_img_source
            if geom_filter != "None":
                preview_img = ModelEngine.transform_geometry(preview_img, geom_filter)
            if warm_val != 0.0 or energy_val != 0.0:
                preview_img = MoodTransformer.transform_image(preview_img, warm_val, energy_val)

            st.image(preview_img, use_column_width=True)

            if apply_btn:
                st.session_state["active_scene_image"] = preview_img
                st.toast("Applied optical grading to active scene canvas!", icon="🎨")
                st.rerun()
        else:
            st.info("Generate or upload an image in Tab 1 to transform its style here.")


# =========================================================================
# TAB 4: EMOTIONAL JOURNEY & STORYBOOK PDF (Timeline & Archives)
# =========================================================================
with tab_journey:
    st.markdown("#### 📈 Emotional Journey Arc & Collectible Storybook")
    st.caption("Explore your session's emotional trajectory across scenes and export your illustrated storybook.")

    history_scenes = st.session_state.get("story_history", [])

    # Interactive Plotly Timeline Chart (Feature 15)
    timeline_fig = TimelineVisualizer.create_emotion_timeline_chart(history_scenes)
    st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    j_col1, j_col2 = st.columns(2, gap="large")

    with j_col1:
        st.markdown("##### 📄 Export Printable Storybook PDF")
        st.caption("Compiles all generated chapters, hero lore, and emotional settings into an illustrated PDF document.")

        if history_scenes:
            pdf_data = ExportEngine.generate_storybook_pdf(
                session_title=st.session_state.get("active_scene_data", {}).get("title", "Dastaan Storybook"),
                character=st.session_state.get("character", DEFAULT_CHARACTERS[0]),
                setting=st.session_state.get("setting", DEFAULT_SETTINGS[0]),
                scenes=history_scenes
            )
            st.download_button(
                label="📄 Download Storybook PDF",
                data=pdf_data,
                file_name="dastaan_living_storybook.pdf",
                mime="application/pdf"
            )
        else:
            st.button("📄 Download Storybook PDF", disabled=True, help="Weave at least 1 chapter in Tab 1 to export your storybook.")

    with j_col2:
        st.markdown("##### 💾 Save to Memory Wall")
        st.caption("Persist this creative journey to the local SQLite database for future replays.")

        if st.button("💾 Archive Current Journey"):
            if history_scenes:
                sess_t = history_scenes[0].get("title", "Creative Session")
                MemoryStore.save_session(
                    session_title=sess_t,
                    character_name=st.session_state.get("character", DEFAULT_CHARACTERS[0])["name"],
                    setting_name=st.session_state.get("setting", DEFAULT_SETTINGS[0])["name"],
                    language=st.session_state.get("language", "English"),
                    scenes=history_scenes
                )
                st.toast("Saved journey to Memory Wall!", icon="💾")
            else:
                st.warning("Generate at least 1 scene in Tab 1 before archiving.")
