import requests
import io
from PIL import Image
import urllib.parse

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
        # Fallback placeholder
        return Image.new("RGB", (800, 800), (40, 40, 60))