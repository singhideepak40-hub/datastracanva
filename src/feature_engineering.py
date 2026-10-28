import numpy as np
from PIL import Image

class FeatureEngineer:
    @staticmethod
    def extract_visual_features(image: Image.Image) -> dict:
        img_np = np.array(image.convert("RGB"))
        gray = np.mean(img_np, axis=2)
        brightness = float(np.mean(gray))
        
        # Calculate saturation
        hsv = np.array(image.convert("HSV"))
        saturation = float(np.mean(hsv[:, :, 1]))
        
        # Edge density approximation
        edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        edge_density = float(min(100.0, edges * 2.5))
        
        return {
            "brightness": brightness,
            "saturation": saturation,
            "edge_density": edge_density
        }