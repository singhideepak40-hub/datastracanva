import os
import sys
import requests
import io
import urllib.parse
from PIL import Image

# Ensure root directory is accessible
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from emotion import get_api_key, CANDIDATE_MODELS
from visuals import VisualSynthesizer

class ImageGenerator:
    @staticmethod
    def enrich_prompt_with_llm(user_prompt: str) -> str:
        """Uses neural intelligence to elevate user prompt into cinematic 8k detail."""
        api_key = get_api_key()
        if not api_key:
            return user_prompt

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
            instruction = (
                "You are an expert AI prompt engineer and cinematographer. "
                "Take this short image prompt and rewrite it into a single highly-detailed, stunning artistic prompt "
                "(around 35-50 words) specifying lighting, mood, camera framing, texture, and 8k detail. "
                "Respond with ONLY the expanded prompt text, nothing else.\n\n"
                f"User Prompt: {user_prompt}"
            )
            payload = {
                "contents": [{"parts": [{"text": instruction}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 100}
            }
            res = requests.post(url, json=payload, timeout=6)
            if res.status_code == 200:
                expanded = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if len(expanded) > 10:
                    return expanded
        except Exception:
            pass

        return user_prompt

    @staticmethod
    def generate_from_prompt(prompt: str, enrich: bool = True) -> Image.Image:
        """Generates high-definition artwork without any watermarks or third-party branding."""
        final_prompt = prompt
        if enrich:
            final_prompt = ImageGenerator.enrich_prompt_with_llm(prompt)

        try:
            encoded_prompt = urllib.parse.quote(final_prompt.replace("\n", " ").strip())
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
            response = requests.get(url, timeout=25)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

        # Generative procedural fallback if offline or timed out
        return VisualSynthesizer.render_procedural_fallback("wonder", prompt)
