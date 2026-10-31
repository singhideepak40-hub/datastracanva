"""
================================================================================
✨ SUDARSHAN STUDIO | Intelligent Visual System & Story Studio ✨
================================================================================
A high-aesthetic creative platform combining Sudarshan's studio precision
with deep multimodal image storytelling and prompt-guided image enhancement.

Core Capabilities:
1. 🔮 Vision Context & Storyteller: Deep context, mood, conveying message & living story
2. 🪄 Prompt Image Enhancer: Transform and enhance the EXACT uploaded image based on prompts
3. ✂️ Magic Studio Cutout: Subject isolation, transparent PNG, DSLR bokeh blur & custom backdrops
4. 🎚️ Mood & Style Transformer: Optical temperature grading & 6 cinematic artistic filters
================================================================================
"""

import os
import sys
import io
import streamlit as st
import numpy as np
from PIL import Image

# Workspace imports
sys.path.insert(0, os.path.dirname(__file__))

from emotion import ImageVisionAnalyzer, PromptImageEnhancer, EMOTION_CONFIG
from visuals import VisualSynthesizer, MoodTransformer
from audio import AudioManager
from src.background_remover import BackgroundRemover
from src.feature_engineering import FeatureEngineer
from src.model_engine import ModelEngine
from src.image_generator import ImageGenerator

# =========================================================================
# PAGE SETUP & SUDARSHAN STUDIO DARK THEME
# =========================================================================
st.set_page_config(
    page_title="Sudarshan Studio | AI Creative Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Dark Obsidian Background */
    .stApp {
        background-color: #0A0D14;
        color: #F0F6FC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Tab Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
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

    /* Highlight Pill Card */
    .info-card {
        background: rgba(18, 24, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .info-card h4 {
        margin: 0 0 8px 0;
        font-size: 15px;
        font-weight: 600;
        color: #58A6FF;
    }
    .info-card p {
        margin: 0;
        font-size: 14px;
        line-height: 1.6;
        color: #C9D1D9;
    }

    /* Story Card */
    .story-frame {
        background: rgba(22, 29, 44, 0.85);
        border-left: 4px solid #58A6FF;
        padding: 20px;
        border-radius: 0 14px 14px 0;
        font-size: 15px;
        line-height: 1.7;
        color: #E6EDF3;
        margin-top: 10px;
        margin-bottom: 16px;
    }

    /* Action Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(180deg, #1A2333 0%, #121824 100%);
        color: #E6EDF3;
        font-weight: 600;
        padding: 10px 18px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #58A6FF;
        background: linear-gradient(180deg, #25334D 0%, #182234 100%);
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SESSION STATE INITIALIZATION
# =========================================================================
if "current_image" not in st.session_state:
    # Default evocative starting canvas so the app is never empty
    st.session_state["current_image"] = VisualSynthesizer.render_procedural_fallback("wonder")

if "vision_insights" not in st.session_state:
    st.session_state["vision_insights"] = {
        "title": "The Awakening of Twilight Horizon",
        "context": "A serene celestial landscape with undulating harmonic waves, starlight points, and deep indigo twilight.",
        "mood": "Calm, mystical, and filled with contemplative wonder.",
        "conveying": "Expresses the vastness of human imagination and the quiet beauty before dawn.",
        "story": (
            "Beneath an obsidian sky bathed in sapphire light, time ceases to rush. "
            "Every ripple on the canvas holds the memory of an unspoken wish, glowing softly "
            "like embers in the quiet hours. In this stillness, a new world begins to stir."
        ),
        "emotion": "wonder",
        "valence": 0.6,
        "arousal": 0.3
    }


# =========================================================================
# HEADER & GLOBAL CANVAS UPLOADER
# =========================================================================
head_col1, head_col2 = st.columns([3, 1])

with head_col1:
    st.markdown("## ✨ Sudarshan Studio")
    st.markdown(
        "<p style='color: #8B949E; margin-top: -14px; font-size: 14px;'>"
        "Studio Visual Intelligence: <b>Deep Image Lore & Context</b> + <b>Prompt-Guided Image Enhancement</b>"
        "</p>",
        unsafe_allow_html=True
    )

with head_col2:
    selected_language = st.selectbox(
        "🌐 Story Language",
        ["Hinglish", "Hindi", "English"],
        index=0,
        help="Language used for context analysis and storytelling."
    )

# Unified Studio Image Uploader (Controls all tabs)
upload_col1, upload_col2 = st.columns([2.5, 1], gap="medium")

with upload_col1:
    uploaded_file = st.file_uploader(
        "📸 Upload Any Image (PNG, JPG, WEBP) to Analyze, Enhance & Transform across all Studio Tools:",
        type=["png", "jpg", "jpeg", "webp"],
        key="master_uploader"
    )
    if uploaded_file is not None:
        try:
            uploaded_img = Image.open(uploaded_file)
            st.session_state["current_image"] = uploaded_img
        except Exception as e:
            st.error(f"Error loading image: {e}")

with upload_col2:
    if st.session_state.get("current_image") is not None:
        feats = FeatureEngineer.extract_visual_features(st.session_state["current_image"])
        st.markdown(
            f"""
            <div style="background: rgba(18, 24, 38, 0.6); padding: 10px 14px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); font-size: 12px; margin-top: 24px;">
              <span style="color: #58A6FF; font-weight: bold;">📊 Spectrum:</span> 
              Bright: <b>{feats['brightness']:.0f}</b> | Sat: <b>{feats['saturation']:.0f}</b> | Detail: <b>{feats['edge_density']:.0f}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# =========================================================================
# THE 5 DEDICATED STUDIO TABS
# =========================================================================
tab_vision, tab_generator, tab_enhancer, tab_studio, tab_transformer = st.tabs([
    "🔮 Vision Context & Storyteller",
    "🎨 AI Image Generator",
    "🪄 Prompt Image Enhancer",
    "✂️ Magic Studio Cutout",
    "🎚️ Mood & Style Transformer"
])


# =========================================================================
# TAB 1: VISION CONTEXT & LIVING STORYTELLER
# =========================================================================
with tab_vision:
    v_left, v_right = st.columns([1.1, 1.4], gap="large")

    with v_left:
        st.markdown("##### 🖼️ Uploaded Canvas")
        active_img = st.session_state.get("current_image")
        if active_img:
            st.image(active_img, use_column_width=True)

        if st.button("🔍 Deep Analyze Context, Mood & Weave Story", key="btn_deep_analyze"):
            with st.spinner("AI Vision is inspecting context, mood and unspoken narrative..."):
                insights = ImageVisionAnalyzer.analyze_uploaded_image(active_img, language=selected_language)
                st.session_state["vision_insights"] = insights
                st.toast("Completed deep visual analysis & story weaving!", icon="✨")
                st.rerun()

    with v_right:
        insights = st.session_state.get("vision_insights", {})

        st.markdown(f"### {insights.get('title', 'Visual Synthesis')}")

        # 1. Context breakdown
        st.markdown(
            f"""
            <div class="info-card">
              <h4>📍 Image Context (Kya Dikh Raha Hai)</h4>
              <p>{insights.get('context', 'Analyzing visual elements and setting...')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Mood & Atmospheric tone
        st.markdown(
            f"""
            <div class="info-card">
              <h4>🎭 Mood & Atmosphere (Vibe Aur Ehsaas)</h4>
              <p>{insights.get('mood', 'Analyzing atmospheric mood and lighting...')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 3. What it is conveying
        st.markdown(
            f"""
            <div class="info-card">
              <h4>💡 What It Conveys (Yeh Kya Express Kar Rahi Hai)</h4>
              <p>{insights.get('conveying', 'Analyzing the core message and symbolic meaning...')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4. Living Story & Lore
        st.markdown("##### 📖 The Story (Kahani):")
        st.markdown(
            f"""
            <div class="story-frame">
              {insights.get('story', 'A story woven directly from this image will appear here.')}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Natural Speech Narration Player
        cur_emo = insights.get("emotion", "wonder")
        cfg = EMOTION_CONFIG.get(cur_emo, EMOTION_CONFIG["neutral"])
        AudioManager.render_audio_controller(
            narration_text=insights.get("story", ""),
            emotion_name=cur_emo,
            sound_type=cfg.get("ambient_sound_type", "forest_breeze"),
            language=selected_language
        )


# =========================================================================
# TAB 2: AI IMAGE GENERATOR (Text-to-Image Masterpiece Synthesis)
# =========================================================================
with tab_generator:
    st.markdown("#### 🎨 AI Image Generator")
    st.caption("Synthesize original, watermark-free high-resolution artwork from pure text prompts.")

    gen_col1, gen_col2 = st.columns([1, 1], gap="large")

    with gen_col1:
        st.markdown("##### Prompt Studio")
        prompt_text = st.text_area(
            "Enter your imaginative vision:",
            value="The Taj Mahal at breathtaking sunset, golden hour lighting reflecting off white marble and reflecting pool, highly detailed 8k photography",
            height=100,
            key="input_gen_prompt"
        )

        use_neural_elevation = st.checkbox(
            "⚡ Neural Prompt Elevation (Enrich with in-house LLM for cinematic lighting & octane details)",
            value=True,
            key="cb_neural_elevation"
        )

        btn_generate = st.button("✨ Synthesize Masterpiece", key="btn_run_text2img")

    with gen_col2:
        st.markdown("##### Synthesized Masterpiece")
        if btn_generate:
            if prompt_text.strip():
                with st.spinner("Neural Vision Engine synthesizing high-definition artwork..."):
                    gen_img = ImageGenerator.generate_from_prompt(prompt_text, enrich=use_neural_elevation)
                    st.session_state["generated_art"] = gen_img
                    st.session_state["generated_prompt_text"] = prompt_text
                    st.toast("Synthesized masterpiece!", icon="🎨")
            else:
                st.warning("Please type a prompt on the left to generate art.")

        displayed_art = st.session_state.get("generated_art")
        if displayed_art:
            st.image(displayed_art, use_column_width=True)
            st.caption(f"Prompt: {st.session_state.get('generated_prompt_text', prompt_text)}")

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                buf = io.BytesIO()
                displayed_art.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Art (PNG)",
                    data=buf.getvalue(),
                    file_name="ai_synthesized_masterpiece.png",
                    mime="image/png"
                )
            with b_col2:
                if st.button("📥 Send to Studio Canvas", key="btn_send_to_canvas"):
                    st.session_state["current_image"] = displayed_art
                    st.toast("Loaded as active canvas for Cutout, Mood & Vision Lore!", icon="📥")
                    st.rerun()
        else:
            st.info("👈 Enter a prompt on the left and click **Synthesize Masterpiece** to create art here.")


# =========================================================================
# TAB 2: PROMPT IMAGE ENHANCER (Transform EXACT Image with Prompts)
# =========================================================================
with tab_enhancer:
    st.markdown("#### 🪄 Prompt-Guided Image Enhancer")
    st.caption("Enter any instruction to enhance and transform the **exact image already uploaded** (lighting, color grading, mood, details).")

    p_col1, p_col2 = st.columns([1, 1], gap="large")

    active_img = st.session_state.get("current_image")

    with p_col1:
        st.markdown("##### Original Image")
        if active_img:
            st.image(active_img, use_column_width=True)

        user_prompt = st.text_input(
            "How do you want to enhance or transform this image?",
            placeholder="e.g. Add golden hour sunset lighting, high contrast and warm glow / Make it cyberpunk neon night / Sharpen details with cinematic color grading",
            key="input_enhancer_prompt"
        )

        enhance_btn = st.button("✨ Enhance & Transform This Image", key="btn_apply_prompt_enhancement")

    with p_col2:
        st.markdown("##### Enhanced Masterpiece")
        if enhance_btn:
            if user_prompt.strip():
                with st.spinner("Transforming and enhancing your exact image..."):
                    res = PromptImageEnhancer.enhance_image_with_prompt(active_img, user_prompt)
                    st.session_state["enhanced_image"] = res["enhanced_image"]
                    st.session_state["enhancement_summary"] = res["summary"]
                    st.toast("Image enhanced successfully!", icon="🪄")
            else:
                st.warning("Please type an enhancement instruction or prompt above.")

        enhanced_img = st.session_state.get("enhanced_image")
        if enhanced_img:
            st.image(enhanced_img, use_column_width=True)
            st.success(f"**Changes:** {st.session_state.get('enhancement_summary', 'Enhanced based on prompt.')}")

            # Download Enhanced Button
            buf = io.BytesIO()
            enhanced_img.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Enhanced Image (PNG)",
                data=buf.getvalue(),
                file_name="enhanced_masterpiece.png",
                mime="image/png"
            )
        else:
            st.info("👈 Enter a prompt on the left and click **Enhance & Transform** to view the result here.")


# =========================================================================
# TAB 4: MAGIC STUDIO CUTOUT (Sudarshan Signature Background Isolation)
# =========================================================================
with tab_studio:
    st.markdown("#### ✂️ Sudarshan Studio Cutout & Background Isolation")
    st.caption("Deep salient object matting with sub-pixel edge feathering & customizable studio backdrops.")

    active_img = st.session_state.get("current_image")

    s_col1, s_col2 = st.columns([1, 1], gap="large")

    with s_col1:
        st.markdown("##### Original Subject")
        if active_img:
            st.image(active_img, use_column_width=True)

        engine_choice = st.selectbox(
            "AI Cutout Precision Engine",
            [
                "🎯 Studio AI (Ultra-Clean Hair & Edges)",
                "⚡ Fast AI (Instant Mobile)",
                "✂️ Classical GrabCut"
            ],
            key="cutout_engine_choice"
        )

        feather_opt = st.checkbox(
            "✨ Sub-Pixel Edge Feathering (Smooth contours & de-fringing)",
            value=True,
            key="cb_feather_opt"
        )

        bg_mode = st.selectbox(
            "Select Studio Backdrop Style",
            [
                "Transparent",
                "DSLR Bokeh Blur",
                "Solid Color",
                "Studio Dark Gradient",
                "Warm Sunset Gradient"
            ],
            key="cutout_bg_mode"
        )

        blur_val = 22
        if bg_mode == "DSLR Bokeh Blur":
            blur_val = st.slider("Bokeh Blur Intensity", 5, 45, 22, 1, key="slider_blur_radius")

        solid_pick = (255, 255, 255)
        if bg_mode == "Solid Color":
            picked_hex = st.color_picker("Pick Studio Backdrop Color", "#FFFFFF", key="cutout_color_picker")
            solid_pick = tuple(int(picked_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        cut_btn = st.button("✂️ Extract Subject & Apply Backdrop", key="btn_apply_cutout")

    with s_col2:
        st.markdown("##### Studio Cutout Result")
        if cut_btn or "studio_cutout_result" in st.session_state:
            if cut_btn:
                with st.spinner("Neural Matting isolating subject with sub-pixel precision..."):
                    raw_cut = BackgroundRemover.remove_background(
                        active_img,
                        engine=engine_choice,
                        feather_edges=feather_opt
                    )
                    comp_res = BackgroundRemover.composite_cutout(
                        raw_cut,
                        active_img,
                        bg_style=bg_mode,
                        solid_color=solid_pick,
                        blur_radius=blur_val
                    )
                    st.session_state["studio_cutout_result"] = comp_res
                    st.toast("Extracted studio cutout with perfect edges!", icon="✂️")

            disp_img = st.session_state.get("studio_cutout_result")
            if disp_img:
                st.image(disp_img, use_column_width=True)

                c_down1, c_down2 = st.columns(2)
                with c_down1:
                    buf = io.BytesIO()
                    disp_img.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Cutout (PNG)",
                        data=buf.getvalue(),
                        file_name="sudarshan_studio_cutout.png",
                        mime="image/png"
                    )
                with c_down2:
                    if st.button("📥 Set as Active Canvas", key="btn_cutout_to_canvas"):
                        st.session_state["current_image"] = disp_img
                        st.toast("Cutout set as active canvas!", icon="📥")
                        st.rerun()
        else:
            st.info("👈 Select your precision engine and backdrop on the left, then click **Extract Subject**.")


# =========================================================================
# TAB 4: MOOD & STYLE TRANSFORMER (Optical Grading & Artistic Filters)
# =========================================================================
with tab_transformer:
    st.markdown("#### 🎚️ Mood & Optical Style Transformer")
    st.caption("Apply real-time cinematic color temperature, dynamic contrast, and artistic style presets.")

    active_img = st.session_state.get("current_image")

    m_col1, m_col2 = st.columns([1, 1.2], gap="large")

    with m_col1:
        st.markdown("##### Artistic Preset Filters")
        style_preset = st.selectbox(
            "Choose Style Preset",
            [
                "None",
                "🎬 Cinematic Teal & Orange",
                "🌅 Golden Hour Sunset",
                "🕶️ Moody Film Noir",
                "👾 Cyberpunk Neon",
                "🎨 Impressionist Oil Painting",
                "✏️ Fine Pencil Sketch"
            ]
        )

        st.markdown("##### Fine-Tune Sliders")
        warmth = st.slider("Temperature: ❄️ Cold vs ☀️ Warm", -1.0, 1.0, 0.0, 0.1)
        dynamics = st.slider("Dynamics: 🌊 Soft vs ⚡ Sharp", -1.0, 1.0, 0.0, 0.1)

        apply_grade = st.button("🎨 Apply Style to Canvas", key="btn_apply_grade")

    with m_col2:
        st.markdown("##### Transformed Preview")
        if active_img:
            preview = active_img
            if style_preset != "None":
                preview = ModelEngine.apply_preset_style(preview, style_preset)
            if warmth != 0.0 or dynamics != 0.0:
                preview = MoodTransformer.transform_image(preview, warmth, dynamics)

            st.image(preview, use_column_width=True)

            if apply_grade:
                st.session_state["current_image"] = preview
                st.toast("Saved styled image as active canvas!", icon="🎨")
                st.rerun()

            # Download Styled Image
            buf = io.BytesIO()
            preview.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Styled Masterpiece (PNG)",
                data=buf.getvalue(),
                file_name="sudarshan_styled_canvas.png",
                mime="image/png"
            )
