import os
import sys
import re
import time
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
    def generate_from_prompt(prompt: str, enrich: bool = True) -> tuple:
        """
        Generates high-definition artwork without any watermarks or third-party branding.
        Handles API cooldowns with automatic retry and graceful feedback.
        Returns: (image: Image.Image, is_real_ai: bool, message: str)
        """
        final_prompt = prompt
        if enrich:
            final_prompt = ImageGenerator.enrich_prompt_with_llm(prompt)

        clean_prompt = final_prompt.replace("\n", " ").strip()
        encoded_prompt = urllib.parse.quote(clean_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

        # Attempt 1: Direct Generation (60s timeout for full diffusion sampling)
        try:
            response = requests.get(url, headers=headers, timeout=60)
            if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                return img, True, "Masterpiece synthesized successfully!"

            # If rate-limited (402 cooldown per IP)
            if response.status_code == 402:
                wait_sec = 22
                try:
                    match = re.search(r"Retry after (\d+)s", response.text)
                    if match:
                        wait_sec = min(int(match.group(1)) + 1, 30)
                except Exception:
                    pass

                time.sleep(wait_sec)

                # Attempt 2: Retry after cooldown
                r2 = requests.get(url, headers=headers, timeout=60)
                if r2.status_code == 200 and "image" in r2.headers.get("content-type", ""):
                    img = Image.open(io.BytesIO(r2.content)).convert("RGB")
                    return img, True, "Masterpiece synthesized successfully!"
        except Exception:
            pass

        # Generative procedural fallback if network timed out
        fallback = VisualSynthesizer.render_procedural_fallback("wonder", prompt)
        return fallback, False, "The public GPU network is temporarily in cooldown (1 request per 45s). Showing procedural preview. Please click Synthesize again in a moment."
