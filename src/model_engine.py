import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

class ModelEngine:
    @staticmethod
    def apply_preset_style(image: Image.Image, preset_name: str) -> Image.Image:
        """Applies advanced cinematic and artistic styles to the image."""
        img = image.convert("RGB")
        img_np = np.array(img).astype(np.float32)

        if preset_name == "🎬 Cinematic Teal & Orange":
            # Boost red/orange in highlights and teal in shadows
            img_np[:, :, 0] = np.clip(img_np[:, :, 0] * 1.15 + 10, 0, 255)  # Red
            img_np[:, :, 1] = np.clip(img_np[:, :, 1] * 1.05, 0, 255)       # Green
            img_np[:, :, 2] = np.clip(img_np[:, :, 2] * 0.90 + (255 - img_np[:, :, 0]) * 0.15, 0, 255) # Blue
            res = Image.fromarray(img_np.astype(np.uint8))
            res = ImageEnhance.Contrast(res).enhance(1.25)
            return ImageEnhance.Color(res).enhance(1.2)

        elif preset_name == "🕶️ Moody Film Noir":
            gray = ImageOps.grayscale(img).convert("RGB")
            contrast = ImageEnhance.Contrast(gray).enhance(1.4)
            return ImageEnhance.Brightness(contrast).enhance(0.9)

        elif preset_name == "🌅 Golden Hour Sunset":
            img_np[:, :, 0] = np.clip(img_np[:, :, 0] * 1.25 + 15, 0, 255)  # Warm reds
            img_np[:, :, 1] = np.clip(img_np[:, :, 1] * 1.10 + 5, 0, 255)   # Amber yellows
            img_np[:, :, 2] = np.clip(img_np[:, :, 2] * 0.80 - 10, 0, 255)  # Cool down blues
            res = Image.fromarray(img_np.astype(np.uint8))
            res = ImageEnhance.Contrast(res).enhance(1.15)
            return ImageEnhance.Color(res).enhance(1.3)

        elif preset_name == "👾 Cyberpunk Neon":
            img_np[:, :, 0] = np.clip(img_np[:, :, 0] * 1.25 + 20, 0, 255)  # Magenta/Pink
            img_np[:, :, 2] = np.clip(img_np[:, :, 2] * 1.35 + 25, 0, 255)  # Electric Blue
            res = Image.fromarray(img_np.astype(np.uint8))
            res = ImageEnhance.Contrast(res).enhance(1.35)
            return ImageEnhance.Color(res).enhance(1.5)

        elif preset_name == "🎨 Impressionist Oil Painting":
            # Edge-preserving bilateral filter + vibrant color
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            filtered = cv2.bilateralFilter(bgr, 9, 75, 75)
            rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            oil = Image.fromarray(rgb)
            return ImageEnhance.Color(oil).enhance(1.35)

        elif preset_name == "✏️ Fine Pencil Sketch":
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            return Image.fromarray(sketch).convert("RGB")

        return img

    @staticmethod
    def transform_geometry(image: Image.Image, filter_name: str) -> Image.Image:
        """Applies geometric and spatial optical filters."""
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
            return image.filter(ImageFilter.GaussianBlur(radius=3))
        return image
