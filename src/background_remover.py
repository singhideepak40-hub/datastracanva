import cv2
import numpy as np
from PIL import Image, ImageFilter

class BackgroundRemover:
    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        """Enhanced subject segmentation using GrabCut refinement returning RGBA cutout."""
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))

        try:
            cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

            img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
            img_rgba[:, :, 3] = mask2 * 255
            return Image.fromarray(cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA))
        except Exception:
            return image.convert("RGBA")

    @staticmethod
    def composite_cutout(
        cutout: Image.Image,
        original: Image.Image,
        bg_style: str = "Transparent",
        solid_color: tuple = (255, 255, 255),
        blur_radius: int = 16
    ) -> Image.Image:
        """Composites foreground cutout onto custom backdrops (Transparent, Studio Solid, DSLR Bokeh Blur, or Gradient)."""
        w, h = cutout.size

        if bg_style == "Transparent":
            return cutout

        elif bg_style == "Solid Color":
            bg = Image.new("RGBA", (w, h), solid_color + (255,))
            bg.paste(cutout, (0, 0), cutout)
            return bg

        elif bg_style == "DSLR Bokeh Blur":
            # Blur the original image heavily and composite sharp cutout on top
            blurred_bg = original.convert("RGBA").filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_bg.paste(cutout, (0, 0), cutout)
            return blurred_bg

        elif bg_style == "Cyberpunk Studio":
            # Gradient from deep navy to vibrant magenta
            bg = Image.new("RGBA", (w, h), (15, 12, 35, 255))
            grad_np = np.zeros((h, w, 4), dtype=np.uint8)
            for y in range(h):
                factor = y / h
                r = int(15 * (1 - factor) + 90 * factor)
                g = int(12 * (1 - factor) + 20 * factor)
                b = int(45 * (1 - factor) + 85 * factor)
                grad_np[y, :, :] = [r, g, b, 255]
            grad_img = Image.fromarray(grad_np)
            grad_img.paste(cutout, (0, 0), cutout)
            return grad_img

        return cutout
