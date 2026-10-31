import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Global cached sessions for instant sub-second inference
CACHED_SESSIONS = {}

def get_rembg_session(model_name: str = "isnet-general-use"):
    """Caches ONNX inference sessions for zero-latency execution."""
    global CACHED_SESSIONS
    if model_name not in CACHED_SESSIONS:
        try:
            import rembg
            CACHED_SESSIONS[model_name] = rembg.new_session(model_name)
        except Exception:
            return None
    return CACHED_SESSIONS.get(model_name)

class BackgroundRemover:
    @staticmethod
    def remove_background(
        image: Image.Image,
        engine: str = "⚡ Fast AI (Instant Studio Matting)",
        feather_edges: bool = True
    ) -> Image.Image:
        """
        State-of-the-art background removal pipeline optimized for cloud environments.
        Supports:
        - U2NetP (ultra-fast lightweight AI, zero-latency, cloud-safe)
        - ISNet (deep salient object matting with memory guard)
        - GrabCut (classical OpenCV morphological fallback)
        """
        img_rgb = image.convert("RGB")
        orig_w, orig_h = img_rgb.size

        # Memory safeguard: cap resolution to 1024px during neural matting to avoid container OOM
        max_dim = 1024
        needs_resize = max(orig_w, orig_h) > max_dim
        if needs_resize:
            scale = max_dim / max(orig_w, orig_h)
            work_img = img_rgb.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS)
        else:
            work_img = img_rgb

        # 1. Fast AI Engine (U2NetP - 4.7MB lightweight model, optimal for cloud RAM)
        if "Fast AI" in engine or "U2NetP" in engine or "Instant" in engine:
            try:
                import rembg
                sess = get_rembg_session("u2netp")
                if sess:
                    res = rembg.remove(work_img, session=sess)
                    if needs_resize:
                        res = res.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                    if feather_edges:
                        res = BackgroundRemover._feather_alpha(res)
                    return res
            except Exception:
                pass

        # 2. Studio AI Engine (ISNet - deep salient object matting, memory-safe)
        if "Studio AI" in engine or "ISNet" in engine or "Ultra-Clean" in engine:
            try:
                import rembg
                sess = get_rembg_session("isnet-general-use")
                if sess:
                    # NOTE: alpha_matting is kept False to prevent scipy sparse matrix OOM on 1GB cloud RAM
                    res = rembg.remove(work_img, session=sess, alpha_matting=False)
                    if needs_resize:
                        res = res.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                    if feather_edges:
                        res = BackgroundRemover._feather_alpha(res)
                    return res
            except Exception:
                pass

        # 3. Robust OpenCV GrabCut Fallback
        return BackgroundRemover._grabcut_fallback(img_rgb, feather_edges)

    @staticmethod
    def _feather_alpha(image_rgba: Image.Image, radius: float = 1.0) -> Image.Image:
        """Applies subtle sub-pixel anti-aliasing to alpha channel to eliminate jagged borders."""
        if image_rgba.mode != "RGBA":
            return image_rgba
        r, g, b, a = image_rgba.split()
        # Smooth alpha mask slightly
        a_smooth = a.filter(ImageFilter.GaussianBlur(radius=radius))
        return Image.merge("RGBA", (r, g, b, a_smooth))

    @staticmethod
    def _grabcut_fallback(image: Image.Image, feather_edges: bool = True) -> Image.Image:
        """Classical OpenCV GrabCut contour segmentation fallback."""
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))

        try:
            cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

            img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            img_rgba[:, :, 3] = mask2 * 255
            result = Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA))
            if feather_edges:
                result = BackgroundRemover._feather_alpha(result)
            return result
        except Exception:
            return image.convert("RGBA")

    @staticmethod
    def composite_cutout(
        cutout: Image.Image,
        original: Image.Image,
        bg_style: str = "Transparent",
        solid_color: tuple = (255, 255, 255),
        blur_radius: int = 20
    ) -> Image.Image:
        """Composites foreground cutout onto studio backdrops (Bokeh blur, solid color, or gradients)."""
        w, h = cutout.size

        if bg_style == "Transparent":
            return cutout

        elif bg_style == "Solid Color":
            bg = Image.new("RGBA", (w, h), solid_color + (255,))
            bg.paste(cutout, (0, 0), cutout)
            return bg

        elif bg_style == "DSLR Bokeh Blur":
            blurred_bg = original.convert("RGBA").filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_bg.paste(cutout, (0, 0), cutout)
            return blurred_bg

        elif bg_style == "Studio Dark Gradient":
            # Gradient from deep graphite to dark midnight
            bg = Image.new("RGBA", (w, h), (18, 22, 30, 255))
            grad_np = np.zeros((h, w, 4), dtype=np.uint8)
            for y in range(h):
                factor = y / h
                r = int(14 * (1 - factor) + 28 * factor)
                g = int(18 * (1 - factor) + 34 * factor)
                b = int(26 * (1 - factor) + 48 * factor)
                grad_np[y, :, :] = [r, g, b, 255]
            grad_img = Image.fromarray(grad_np)
            grad_img.paste(cutout, (0, 0), cutout)
            return grad_img

        elif bg_style == "Warm Sunset Gradient":
            grad_np = np.zeros((h, w, 4), dtype=np.uint8)
            for y in range(h):
                factor = y / h
                r = int(255 * (1 - factor) + 180 * factor)
                g = int(140 * (1 - factor) + 80 * factor)
                b = int(90 * (1 - factor) + 120 * factor)
                grad_np[y, :, :] = [r, g, b, 255]
            grad_img = Image.fromarray(grad_np)
            grad_img.paste(cutout, (0, 0), cutout)
            return grad_img

        return cutout
