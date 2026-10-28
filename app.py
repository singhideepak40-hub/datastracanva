import streamlit as st
import os
import io
import random
import urllib.parse
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import requests

st.set_page_config(
    page_title="MagickImg | AI Image Story Generator",
    page_icon="✨",
    layout="wide"
)


# ==========================================
# 1. DATA LOADER MODULE
# ==========================================
class DataLoader:
    @staticmethod
    def load_image_from_upload(uploaded_file) -> Image.Image:
        try:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            bytes_data = uploaded_file.read()
            img = Image.open(io.BytesIO(bytes_data))
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                return img.convert("RGBA")
            return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")


# ==========================================
# 2. FEATURE ENGINEERING MODULE
# ==========================================
class FeatureEngineer:
    @staticmethod
    def extract_visual_features(image: Image.Image) -> dict:
        img_np = np.array(image.convert("RGB"))
        gray = np.mean(img_np, axis=2)
        brightness = float(np.mean(gray))

        hsv = np.array(image.convert("HSV"))
        saturation = float(np.mean(hsv[:, :, 1]))

        edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        edge_density = float(min(100.0, edges * 2.5))

        return {
            "brightness": brightness,
            "saturation": saturation,
            "edge_density": edge_density
        }


# ==========================================
# 3. BACKGROUND REMOVER MODULE
# ==========================================
class BackgroundRemover:
    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))

        try:
            cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

            img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            img_rgba[:, :, 3] = mask2 * 255
            return Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA))
        except Exception:
            return image.convert("RGBA")


# ==========================================
# 4. MODEL ENGINE MODULE
# ==========================================
class ModelEngine:
    @staticmethod
    def apply_mood(image: Image.Image, mood: str) -> Image.Image:
        if "Noir" in mood:
            return ImageOps.grayscale(image).convert("RGB")
        elif "Vibrant" in mood:
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(1.4)
        return image

    @staticmethod
    def transform_geometry(image: Image.Image, filter_name: str) -> Image.Image:
        if filter_name == "None" or not filter_name:
            return image
        elif filter_name == "Grayscale Conversion":
            return ImageOps.grayscale(image).convert("RGB")
        elif filter_name == "Invert Colors":
            return ImageOps.invert(image.convert("RGB"))
        elif filter_name == "Sharpen Details":
            return image.filter(ImageFilter.SHARPEN)
        elif filter_name == "Edge Highlight":
            return image.filter(ImageFilter.FIND_EDGES).convert("RGB")
        elif filter_name == "Gaussian Blur":
            return image.filter(ImageFilter.GaussianBlur(radius=2))
        return image


# ==========================================
# 5. STORY GENERATOR MODULE
# ==========================================
class StoryGenerator:
    @staticmethod
    def generate_narrative(mood: str, features: dict) -> dict:
        brightness = features.get("brightness", 128)
        saturation = features.get("saturation", 100)

        if saturation > 120:
            title = "The Realm of Luminous Fables"
            thematic_arc = "A vibrant awakening of color, where every hue tells an ancient secret of light."
            story = (
                "In a dimension where pigments breathe and shadows dance, this snapshot captures the precise heartbeat "
                "of an eternal morning. Sunlight cascades across the canvas like liquid gold, whispering tales of forgotten "
                "kingdoms where the sky is ever-blazing and wonder knows no horizon. Every pixel radiates an unstoppable pulse "
                "of life, inviting travelers to step beyond the veil into pure enchantment."
            )
        elif brightness < 80:
            title = "Chronicles of the Obsidian Veil"
            thematic_arc = "A quiet nocturnal mystery where whispers linger in the velvet dark."
            story = (
                "Beneath a canopy of silent stars and silver moonlight lies a world veiled in deep obsidian shadows. "
                "Here, time slows to a gentle murmur. The darkness is not empty; it is alive with untold secrets, "
                "ancient nocturnal echoes, and the quiet magic of souls who thrive when the rest of the realm is fast asleep."
            )
        else:
            title = "The Alchemist's Timeless Echo"
            thematic_arc = "A delicate equilibrium between light and memory, preserved in stardust."
            story = (
                "A timeless fragment caught between the amber glow of yesterday and the starry promise of tomorrow. "
                "The air shimmers with nostalgic warmth, as though the very atmosphere holds the memories of wanderers "
                "who once walked this enchanted path. A true masterpiece of visual poetry."
            )

        return {
            "title": title,
            "thematic_arc": thematic_arc,
            "story": story,
            "mood_summary": f"Detected Mood: {mood} (Brightness: {brightness:.1f}, Saturation: {saturation:.1f})"
        }


# ==========================================
# 6. IMAGE GENERATOR MODULE
# ==========================================
class ImageGenerator:
    @staticmethod
    def generate_from_prompt(prompt: str) -> Image.Image:
        try:
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            pass
        return Image.new("RGB", (800, 800), (40, 40, 60))


# ==========================================
# 7. STREAMLIT UI APPLICATION
# ==========================================
st.title("✨ MagickImg | AI Image Story Generator")
st.markdown("### 🔮 Transform Any Image into a Magical Narrative, Context Title & Immersive Lore")

with st.sidebar:
    st.header("⚙️ Studio Controls")
    task_mode = st.selectbox(
        "Select Operation Mode",
        ["Dynamic Vision Story & Analysis", "Lightning-Fast Background Removal", "AI Image Generation"]
    )

    geometry_option = st.selectbox(
        "Geometric Filter",
        ["None", "Grayscale Conversion", "Sharpen Details", "Edge Highlight", "Invert Colors", "Gaussian Blur"]
    )

if task_mode in ["Dynamic Vision Story & Analysis", "Lightning-Fast Background Removal"]:
    uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, WEBP)")

    if uploaded_file is not None:
        try:
            image = DataLoader.load_image_from_upload(uploaded_file)
            img_rgb = image.convert("RGB") if image.mode != "RGB" else image

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Canvas")
                st.image(image, use_column_width=True)

            if task_mode == "Lightning-Fast Background Removal":
                with st.spinner("Extracting foreground subject with enhanced precision..."):
                    cutout_img = BackgroundRemover.remove_background(img_rgb)
                with col2:
                    st.subheader("Isolated Subject Cutout")
                    st.image(cutout_img, use_column_width=True)
            else:
                with st.spinner("✨ Weaving magical lore & extracting image context..."):
                    features = FeatureEngineer.extract_visual_features(img_rgb)
                    auto_mood = "Vibrant Energy" if features["saturation"] > 100 else "Melancholic / Noir"
                    graded_img = ModelEngine.apply_mood(img_rgb, auto_mood)
                    graded_img = ModelEngine.transform_geometry(graded_img, geometry_option)
                    story_data = StoryGenerator.generate_narrative(auto_mood, features)

                with col2:
                    st.subheader("Magical Visual Synthesis")
                    st.image(graded_img, use_column_width=True)

                st.markdown("---")
                st.subheader(f"🔮 Predicted Context Title: {story_data['title']}")
                st.success(f"**Detected Mood:** {auto_mood}")
                st.info(story_data["story"])

                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Brightness Index", f"{features['brightness']:.1f}")
                mcol2.metric("Chromatic Saturation", f"{features['saturation']:.1f}")
                mcol3.metric("Edge Density", f"{features['edge_density']:.1f}%")

                st.write(f"**Thematic Lore:** {story_data['thematic_arc']}")

        except Exception as e:
            st.error(f"Processing error: {e}")

else:
    st.subheader("🎨 MagickImg Diffusion Generator")
    prompt_input = st.text_input("Enter your magical prompt:",
                                 "The Taj Mahal at breathtaking sunset, golden hour lighting reflecting off white marble and reflecting pool, highly detailed 8k")
    if st.button("Generate AI Masterpiece"):
        with st.spinner("Synthesizing magical AI art..."):
            gen_image = ImageGenerator.generate_from_prompt(prompt_input)
            st.image(gen_image, caption=f"Prompt: {prompt_input}", use_column_width=True)