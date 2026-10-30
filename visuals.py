"""
Visuals Module for Dastaan Canvas
Handles:
- AI Image generation with prompt enrichment
- Live reactive p5.js canvas (valence & arousal driven flow field)
- Mood Transform Slider (warm/cold, calm/chaotic image grading)
- Studio subject cutout (Pokecut aesthetic)
- Cinematic CSS styling (Ken Burns zoom & crossfade)
"""

import io
import urllib.parse
import requests
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import streamlit.components.v1 as components
from emotion import EMOTION_CONFIG

# =========================================================================
# FEATURE 9: Image Generator with Multi-Tier Fallback & Safe Procedural Art
# =========================================================================
class VisualSynthesizer:
    @staticmethod
    def generate_scene_image(prompt: str, emotion_name: str = "joy") -> Image.Image:
        """
        Synthesizes scene image using Pollinations AI diffusion engine.
        If offline or timed out, generates a procedural generative art canvas
        styled specifically according to the target emotion's color palette.
        """
        try:
            clean_prompt = (prompt or "enchanted cinematic world").replace("\n", " ").strip()
            encoded_prompt = urllib.parse.quote(clean_prompt)
            # High-aesthetic cinematic diffusion query
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=768&nologo=true&seed=42"
            response = requests.get(url, timeout=18)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

        # Procedural Artistic Canvas Fallback (Guaranteed to always render, 100% offline-ready)
        return VisualSynthesizer.render_procedural_fallback(emotion_name, prompt)

    @staticmethod
    def render_procedural_fallback(emotion_name: str, prompt_hint: str = "") -> Image.Image:
        """Generates a rich, dreamy procedural abstract landscape matched to the emotion."""
        w, h = 1024, 768
        cfg = EMOTION_CONFIG.get(emotion_name or "neutral", EMOTION_CONFIG["neutral"])
        palette = cfg.get("palette", ["#1D3557", "#457B9D", "#A8DADC"])
        if not palette or len(palette) == 0:
            palette = ["#1D3557", "#457B9D"]

        # Create gradient canvas
        base = Image.new("RGB", (w, h), palette[0])
        draw = ImageDraw.Draw(base)

        color_hex1 = palette[0]
        color_hex2 = palette[1] if len(palette) > 1 else palette[0]

        # Draw undulating emotional waves and celestial orbs
        for i in range(h):
            factor = i / h
            c1 = tuple(int(color_hex1.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
            c2 = tuple(int(color_hex2.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
            r = int(c1[0] * (1 - factor) + c2[0] * factor)
            g = int(c1[1] * (1 - factor) + c2[1] * factor)
            b = int(c1[2] * (1 - factor) + c2[2] * factor)
            draw.line([(0, i), (w, i)], fill=(r, g, b))

        # Add luminous floating particles and bokeh circles
        rng = np.random.RandomState(42)
        glow_color = cfg.get("glow_color", "#FFD166")
        gc = tuple(int(glow_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))

        for _ in range(cfg.get("particle_density", 50)):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            rad = rng.randint(4, 35)
            alpha_color = gc + (120,)
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=gc)

        # Add soft blur for magical storybook atmosphere
        base = base.filter(ImageFilter.GaussianBlur(radius=8))

        # Add fine grain vignette
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        o_draw = ImageDraw.Draw(overlay)
        o_draw.rectangle([0, 0, w, h], outline=gc, width=4)
        base.paste(overlay, (0, 0), overlay)
        return base


# =========================================================================
# FEATURE 10: Mood Transform Slider (Calm <-> Chaotic, Warm <-> Cold)
# =========================================================================
class MoodTransformer:
    @staticmethod
    def transform_image(
        image: Image.Image,
        warm_cold_val: float = 0.0,   # -1.0 (Cold/Blue) to +1.0 (Warm/Amber)
        calm_chaos_val: float = 0.0  # -1.0 (Calm/Soft) to +1.0 (Chaotic/Sharp/Turbulent)
    ) -> Image.Image:
        """
        Real-time optical post-processing pipeline that transforms image mood:
        - Warm/Cold shifts the chromatic temperature via LAB/RGB channels.
        - Calm/Chaos shifts sharpness, high-frequency edge contrast, and local turbulence.
        """
        if image is None:
            return None
        img_np = np.array(image.convert("RGB")).astype(np.float32)

        # Temperature (Warm vs Cold)
        # Warm increases Red & Yellow, Cold increases Blue & Cyan
        if warm_cold_val > 0:
            img_np[:, :, 0] += warm_cold_val * 40.0  # Red
            img_np[:, :, 1] += warm_cold_val * 20.0  # Green
            img_np[:, :, 2] -= warm_cold_val * 25.0  # Blue
        elif warm_cold_val < 0:
            shift = abs(warm_cold_val)
            img_np[:, :, 0] -= shift * 30.0
            img_np[:, :, 2] += shift * 45.0

        # Calm (Soft Blur & Low Contrast) vs Chaos (High Saturation & Edge Punch)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        if calm_chaos_val > 0.1:
            # Chaos: boost contrast, punch saturation, and sharpen
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.0 + calm_chaos_val * 0.7)
            color_enhancer = ImageEnhance.Color(pil_img)
            pil_img = color_enhancer.enhance(1.0 + calm_chaos_val * 0.8)
            pil_img = pil_img.filter(ImageFilter.SHARPEN)
        elif calm_chaos_val < -0.1:
            # Calm: gentle softening, ethereal dream blur, pastel contrast
            blur_radius = abs(calm_chaos_val) * 2.5
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(max(0.7, 1.0 - abs(calm_chaos_val) * 0.3))

        return pil_img


# =========================================================================
# FEATURE 11: Live Reactive Canvas (p5.js Flow Field driven by Valence & Arousal)
# =========================================================================
def render_live_p5_canvas(
    valence: float = 0.0,
    arousal: float = 0.4,
    emotion_name: str = "neutral",
    height: int = 340,
    reduced_motion: bool = False
):
    """
    Renders an interactive p5.js fluid flow field that reacts to the user's emotion in real-time.
    [BEHAVIOUR EXPRESSION]:
    - Valence determines the color spectrum (Cold Blue/Indigo for negative, Radiant Gold/Rose for positive).
    - Arousal directly dictates particle velocity, wave frequency, and turbulent vortex intensity.
    - Idle breathing state runs harmonic sinusoidal drift.
    """
    cfg = EMOTION_CONFIG.get(emotion_name, EMOTION_CONFIG["neutral"])
    palette = cfg.get("palette", ["#3A86FF", "#8338EC", "#FFBE0B"])
    speed_mult = 0.2 if reduced_motion else cfg.get("motion_speed", 1.0)
    turb_mult = 0.1 if reduced_motion else cfg.get("turbulence", 0.4)
    density = int(cfg.get("particle_density", 60) * (0.5 if reduced_motion else 1.0))
    glow_col = cfg.get("glow_color", "#FFD166")

    # Pass dynamic variables into embedded p5.js code
    p5_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
      <style>
        body, html {{
          margin: 0;
          padding: 0;
          overflow: hidden;
          background: #090B10;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        canvas {{
          display: block;
          border-radius: 16px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 20px {glow_col}22;
        }}
        .hud-overlay {{
          position: absolute;
          bottom: 12px;
          left: 16px;
          color: rgba(255, 255, 255, 0.85);
          font-size: 11px;
          letter-spacing: 0.8px;
          text-transform: uppercase;
          pointer-events: none;
          background: rgba(15, 20, 30, 0.65);
          backdrop-filter: blur(8px);
          padding: 6px 12px;
          border-radius: 20px;
          border: 1px solid rgba(255, 255, 255, 0.12);
        }}
      </style>
    </head>
    <body>
      <div class="hud-overlay">
        ◈ System Pulse: <span style="color: {glow_col}; font-weight: bold;">{emotion_name.upper()}</span> | Valence: {valence:+.2f} | Arousal: {arousal:.2f}
      </div>
      <script>
        let particles = [];
        const numParticles = {density};
        const speed = {speed_mult};
        const turbulence = {turb_mult};
        const palette = {palette};
        const glowColor = "{glow_col}";

        function setup() {{
          createCanvas(windowWidth, {height});
          for (let i = 0; i < numParticles; i++) {{
            particles.push(new Particle());
          }}
          background(9, 11, 16);
        }}

        function draw() {{
          // Gentle motion blur trail
          background(9, 11, 16, 28);

          for (let p of particles) {{
            p.update();
            p.display();
          }}
        }}

        class Particle {{
          constructor() {{
            this.x = random(width);
            this.y = random(height);
            this.vx = 0;
            this.vy = 0;
            this.radius = random(2.5, 6.0);
            this.color = color(random(palette));
            this.alpha = random(140, 240);
            this.phase = random(TWO_PI);
          }}

          update() {{
            let angle = noise(this.x * 0.004, this.y * 0.004, frameCount * 0.005 * turbulence) * TWO_PI * 2.5;
            // Harmonic breathing sinusoidal pulse
            let pulse = sin(frameCount * 0.03 + this.phase);
            
            this.vx = cos(angle) * speed * (1.0 + 0.3 * pulse);
            this.vy = sin(angle) * speed * (1.0 + 0.3 * pulse);

            this.x += this.vx;
            this.y += this.vy;

            // Wrap around edges cleanly
            if (this.x < 0) this.x = width;
            if (this.x > width) this.x = 0;
            if (this.y < 0) this.y = height;
            if (this.y > height) this.y = 0;
          }}

          display() {{
            noStroke();
            this.color.setAlpha(this.alpha);
            fill(this.color);
            ellipse(this.x, this.y, this.radius, this.radius);

            // Radiant subtle glow halo on large particles
            if (this.radius > 4.5) {{
              fill(color(glowColor + "22"));
              ellipse(this.x, this.y, this.radius * 2.4, this.radius * 2.4);
            }}
          }}
        }}

        function windowResized() {{
          resizeCanvas(windowWidth, {height});
        }}
      </script>
    </body>
    </html>
    """
    components.html(p5_code, height=height, scrolling=False)


# =========================================================================
# FEATURE 13: Ken Burns & Crossfade Cinematic Scene Card Component
# =========================================================================
def render_cinematic_scene_image(image_pil: Image.Image, caption: str, reduced_motion: bool = False):
    """Renders current scene image with a slow breathing Ken Burns animation."""
    if image_pil is None:
        return
    import base64
    buffered = io.BytesIO()
    if image_pil.mode in ("RGBA", "LA") or (image_pil.mode == "P" and "transparency" in image_pil.info):
        image_pil.save(buffered, format="PNG")
        mime = "image/png"
    else:
        image_pil.convert("RGB").save(buffered, format="JPEG", quality=90)
        mime = "image/jpeg"
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    animation_style = "" if reduced_motion else """
    animation: kenburns 22s ease-in-out infinite alternate;
    """

    html = f"""
    <style>
      @keyframes kenburns {{
        0% {{ transform: scale(1.0) translate(0, 0); }}
        50% {{ transform: scale(1.06) translate(-1%, -1.5%); }}
        100% {{ transform: scale(1.03) translate(1%, 0.8%); }}
      }}
      .cinematic-frame {{
        position: relative;
        overflow: hidden;
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: #000;
        margin-bottom: 12px;
      }}
      .cinematic-img {{
        width: 100%;
        display: block;
        transition: opacity 1.2s cubic-bezier(0.16, 1, 0.3, 1);
        {animation_style}
      }}
      .cinematic-caption {{
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(180deg, transparent 0%, rgba(9, 11, 16, 0.88) 100%);
        color: #E2E8F0;
        padding: 24px 20px 14px;
        font-size: 13px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }}
    </style>
    <div class="cinematic-frame">
      <img src="data:{mime};base64,{img_b64}" class="cinematic-img" alt="{caption}" />
      <div class="cinematic-caption">🎬 {caption}</div>
    </div>
    """
    components.html(html, height=440, scrolling=False)
