import os
import io
from PIL import Image

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