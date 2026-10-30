"""
Emotion Engine Module for Dastaan Canvas
Handles Text, Voice, Webcam, and Vision-based multimodal emotion extraction.
Maps human feelings to continuous Valence (-1..1), Arousal (0..1), and Intensity (0..1).
"""

import json
import os
import re
import requests
import streamlit as st
import numpy as np
from PIL import Image

# Central emotion mapping config loader
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "emotion_map.json")

def load_emotion_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Fallback dictionary if config is missing
        return {
            "joy": {"valence": 0.8, "arousal": 0.7, "palette": ["#FFB703", "#FB8500", "#FFD166"], "motion_speed": 1.8},
            "sadness": {"valence": -0.8, "arousal": 0.2, "palette": ["#1D3557", "#457B9D", "#A8DADC"], "motion_speed": 0.4},
            "anger": {"valence": -0.7, "arousal": 0.9, "palette": ["#D00000", "#9D0208", "#FF5400"], "motion_speed": 2.2},
            "serenity": {"valence": 0.6, "arousal": 0.2, "palette": ["#2A9D8F", "#E9C46A", "#264653"], "motion_speed": 0.5},
            "wonder": {"valence": 0.7, "arousal": 0.8, "palette": ["#7209B7", "#4361EE", "#4CC9F0"], "motion_speed": 1.4},
            "nostalgia": {"valence": 0.3, "arousal": 0.4, "palette": ["#D4A373", "#CCD5AE", "#FAEDCD"], "motion_speed": 0.7},
            "fear": {"valence": -0.6, "arousal": 0.7, "palette": ["#03071E", "#370617", "#9D0208"], "motion_speed": 1.2},
            "neutral": {"valence": 0.0, "arousal": 0.4, "palette": ["#3A86FF", "#8338EC", "#FFBE0B"], "motion_speed": 1.0}
        }

EMOTION_CONFIG = load_emotion_config()

def get_api_key():
    """Retrieve Gemini API key from Streamlit secrets, env vars, or session state."""
    if "gemini_api_key" in st.session_state and st.session_state["gemini_api_key"]:
        return st.session_state["gemini_api_key"].strip()
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()

CANDIDATE_MODELS = ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-flash-latest"]

# =========================================================================
# FEATURE 1: LLM Emotion Engine with Strict JSON Validation & Fallback
# =========================================================================
class EmotionEngine:
    @staticmethod
    def _call_gemini_json(prompt: str, api_key: str, model: str = None) -> dict:
        """Invokes Gemini REST API enforcing strict JSON schema across candidate models."""
        models_to_try = [model] if model else CANDIDATE_MODELS
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "response_mime_type": "application/json"
            }
        }

        last_err = None
        for m in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
                response = requests.post(url, headers=headers, json=payload, timeout=12)
                if response.status_code == 200:
                    res_data = response.json()
                    raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
                    return json.loads(cleaned)
                else:
                    last_err = f"Model {m} status {response.status_code}: {response.text}"
            except Exception as e:
                last_err = str(e)
                continue

        raise RuntimeError(f"All neural models failed: {last_err}")

    @staticmethod
    def _rule_based_fallback(text: str) -> dict:
        """High-precision local heuristic fallback when offline or no API key."""
        t = text.lower()
        sad_words = ["sad", "crying", "lost", "alone", "dark", "sorrow", "grief", "broken", "rain", "tears", "pain", "chup", "udaas", "dard", "tanhai"]
        joy_words = ["happy", "joy", "excited", "sunshine", "golden", "laugh", "love", "cheer", "play", "celebrate", "khush", "masti", "muskaan"]
        anger_words = ["angry", "rage", "fire", "furious", "hate", "fight", "storm", "burning", "gussa", "nafrat", "krodh"]
        wonder_words = ["wonder", "magic", "portal", "dream", "star", "cosmic", "universe", "mystery", "jadu", "khwab", "hairat"]
        peace_words = ["peace", "calm", "serene", "quiet", "breathe", "ocean", "zen", "still", "sukoon", "shanti", "itminan"]
        nostalgia_words = ["remember", "childhood", "toy", "memory", "attic", "old", "yesterday", "vintage", "yaad", "bachpan"]

        def count_hits(words):
            return sum(1 for w in words if w in t)

        scores = {
            "sadness": count_hits(sad_words),
            "joy": count_hits(joy_words),
            "anger": count_hits(anger_words),
            "wonder": count_hits(wonder_words),
            "serenity": count_hits(peace_words),
            "nostalgia": count_hits(nostalgia_words)
        }

        best_emotion, count = max(scores.items(), key=lambda x: x[1])
        if count == 0:
            best_emotion = "neutral"

        cfg = EMOTION_CONFIG.get(best_emotion, EMOTION_CONFIG["neutral"])
        return {
            "emotion": best_emotion,
            "valence": cfg["valence"],
            "arousal": cfg["arousal"],
            "intensity": 0.6 if count > 0 else 0.4,
            "keywords": [w for w in (sad_words + joy_words + anger_words + wonder_words + peace_words + nostalgia_words) if w in t][:4],
            "reasoning": f"Local Lexical & Sentiment analyzer parsed input containing emotional markers for '{best_emotion}'."
        }

    @classmethod
    def analyze_text(cls, text: str) -> dict:
        """
        Analyzes user text (Hindi, English, Hinglish) and returns normalized emotion data.
        [BEHAVIOUR EXPRESSION]: The detected valence (-1..1) and arousal (0..1) directly drive
        p5.js particle speed, color temperature, and story branching momentum.
        """
        if not text or not text.strip():
            return {
                "emotion": "neutral",
                "valence": 0.0,
                "arousal": 0.4,
                "intensity": 0.3,
                "keywords": ["idle", "breathing"],
                "reasoning": "Idle state: system waiting in gentle breathing equilibrium."
            }

        api_key = get_api_key()
        if not api_key:
            return cls._rule_based_fallback(text)

        prompt = f"""
You are an expert psychological & aesthetic emotion analyzer.
Analyze the following user text input (which may be in English, Hindi, or Hinglish):
"{text}"

Identify the core emotional state. Categorize it strictly into ONE of:
["joy", "sadness", "serenity", "wonder", "anger", "nostalgia", "fear", "neutral"].

Output ONLY a JSON object matching this schema:
{{
  "emotion": "string (one of joy, sadness, serenity, wonder, anger, nostalgia, fear, neutral)",
  "valence": float between -1.0 (very negative) and 1.0 (very positive),
  "arousal": float between 0.0 (calm/low energy) and 1.0 (agitated/high energy),
  "intensity": float between 0.0 and 1.0,
  "keywords": ["list", "of", "emotional", "keywords"],
  "reasoning": "brief 1-sentence explanation of why this emotion was chosen"
}}
"""
        try:
            result = cls._call_gemini_json(prompt, api_key)
            # Enforce range limits
            result["valence"] = float(max(-1.0, min(1.0, result.get("valence", 0.0))))
            result["arousal"] = float(max(0.0, min(1.0, result.get("arousal", 0.5))))
            result["intensity"] = float(max(0.0, min(1.0, result.get("intensity", 0.5))))
            if result.get("emotion") not in EMOTION_CONFIG:
                result["emotion"] = "neutral"
            return result
        except Exception:
            # Automatic graceful fallback
            return cls._rule_based_fallback(text)


# =========================================================================
# FEATURE 2: Optional Webcam Emotion Input (Consent Enforced, Zero Frame Storage)
# =========================================================================
class WebcamEmotionAnalyzer:
    @staticmethod
    def analyze_frame(image: Image.Image) -> dict:
        """
        Analyzes a single camera snapshot strictly in-memory.
        Extracts facial mood / expression cues.
        """
        api_key = get_api_key()
        if api_key:
            # Use Gemini Multimodal Vision API
            try:
                import io
                import base64
                buffered = io.BytesIO()
                image.convert("RGB").save(buffered, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": "Analyze the human face and emotional atmosphere in this photo. "
                                     "Classify into strictly one of: ['joy', 'sadness', 'serenity', 'wonder', 'anger', 'nostalgia', 'fear', 'neutral']. "
                                     "Respond with ONLY valid JSON: {\"emotion\": string, \"valence\": float (-1.0 to 1.0), \"arousal\": float (0.0 to 1.0), \"intensity\": float (0.0 to 1.0), \"keywords\": [string], \"reasoning\": string}"},
                            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                        ]
                    }],
                    "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"}
                }

                for m in CANDIDATE_MODELS:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
                        res = requests.post(url, headers=headers, json=payload, timeout=12)
                        if res.status_code == 200:
                            raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
                            return json.loads(cleaned)
                    except Exception:
                        continue
            except Exception:
                pass

        # Local visual feature fallback for webcam frame
        img_np = np.array(image.convert("RGB"))
        avg_brightness = float(np.mean(img_np))
        if avg_brightness > 140:
            emotion = "joy"
        elif avg_brightness < 70:
            emotion = "sadness"
        else:
            emotion = "serenity"

        cfg = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])
        return {
            "emotion": emotion,
            "valence": cfg["valence"],
            "arousal": cfg["arousal"],
            "intensity": 0.5,
            "keywords": ["expression", "lighting", "tone"],
            "reasoning": f"Webcam optical ambient analysis detected brightness index of {avg_brightness:.1f} correlating with {emotion}."
        }


# =========================================================================
# FEATURE 4: Image Upload Vision Mood Analysis
# =========================================================================
class ImageVisionAnalyzer:
    @staticmethod
    def analyze_uploaded_image(image: Image.Image) -> dict:
        """
        Multimodal LLM vision analysis: extracts subject, color palette, and emotional ambiance.
        """
        api_key = get_api_key()
        if api_key:
            try:
                import io
                import base64
                buffered = io.BytesIO()
                image.convert("RGB").save(buffered, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": "Examine this artwork/photograph as a senior art director. Describe the mood, color palette, subject, and story atmosphere. "
                                     "Output ONLY JSON: {\"emotion\": string (one of joy, sadness, serenity, wonder, anger, nostalgia, fear, neutral), \"valence\": float (-1.0 to 1.0), \"arousal\": float (0.0 to 1.0), \"intensity\": float (0.0 to 1.0), \"palette_summary\": string, \"subject\": string, \"keywords\": [string], \"reasoning\": string}"},
                            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                        ]
                    }],
                    "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"}
                }

                for m in CANDIDATE_MODELS:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
                        res = requests.post(url, headers=headers, json=payload, timeout=14)
                        if res.status_code == 200:
                            raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
                            return json.loads(cleaned)
                    except Exception:
                        continue
            except Exception:
                pass

        # Heuristic fallback using HSV & luminosity
        img_np = np.array(image.convert("RGB"))
        gray = np.mean(img_np, axis=2)
        brightness = float(np.mean(gray))
        hsv = np.array(image.convert("HSV"))
        saturation = float(np.mean(hsv[:, :, 1]))

        if saturation > 125:
            emotion = "joy"
        elif brightness < 80:
            emotion = "sadness"
        elif saturation < 60:
            emotion = "nostalgia"
        else:
            emotion = "serenity"

        cfg = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])
        return {
            "emotion": emotion,
            "valence": cfg["valence"],
            "arousal": cfg["arousal"],
            "intensity": 0.6,
            "palette_summary": f"Chromatic Saturation: {saturation:.1f}, Luminance: {brightness:.1f}",
            "subject": "Captured visual canvas",
            "keywords": ["chroma", "light", "atmosphere"],
            "reasoning": f"Optical spectrum extraction: Saturation {saturation:.1f} and Luminance {brightness:.1f} indicate a {emotion} atmosphere."
        }
