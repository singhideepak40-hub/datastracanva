from PIL import Image, ImageOps, ImageEnhance

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
            from PIL import ImageFilter
            return image.filter(ImageFilter.SHARPEN)
        elif filter_name == "Edge Highlight":
            from PIL import ImageFilter
            return image.filter(ImageFilter.FIND_EDGES).convert("RGB")
        elif filter_name == "Gaussian Blur":
            from PIL import ImageFilter
            return image.filter(ImageFilter.GaussianBlur(radius=2))
        return image