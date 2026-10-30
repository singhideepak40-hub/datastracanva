# ✨ Dastaan Canvas (दास्तान कैनवास)

> **Art that feels what you feel, and weaves a living tale scene by scene.**  
> A proprietary multimodal AI visual system whose **behaviour is the message**—where human emotion dynamically dictates color palettes, fluid particle physics, cinematic framing, acoustic soundscapes, and branching narrative lore.  
> Uniting the dark, high-precision canvas aesthetic of **Pokecut** with the magical, nostalgic wonder of **Toy Story**.

---

## 🌟 3-Line Pitch
1. **Emotional Symbiosis:** Rather than generating isolated images, Dastaan Canvas listens to your emotional coordinates (valence and arousal) via text, voice, camera, or vision analysis.
2. **Behavioral Expression:** System physics—p5.js particle velocity, color temperature, sound harmonics, and narrative tension—visibly and audibly transform to express your state of mind.
3. **Living Storybook:** Persistent heroes (like *Babu the Clockwork Sparrow*) embark on a branching cinematic journey where your choices and emotions steer each chapter.

---

## 🚀 Quick Setup & Installation

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/singhideepak40-hub/datastracanva.git
cd datastracanva
pip install -r requirements.txt
```

### 2. Configure Neural Engine Secrets (Optional)
Neural engine credentials can be stored in `.streamlit/secrets.toml` or via system environment variables:
```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-api-key"
```
*Note: If no API key is supplied, Dastaan Canvas automatically runs its built-in local sentiment heuristics and offline procedural generative art canvas.*

### 3. Launch Application
```bash
streamlit run app.py
```

---

## 🧭 Clean Tabbed Architecture

| Tab | Feature & Capabilities |
| :--- | :--- |
| **🎨 Living Canvas & Story Studio** | Multimodal emotion analysis, dynamic diffusion art, live reactive p5.js fluid canvas, audio narration & branching lore |
| **✂️ Magic Studio Cutout** | Pokecut-style foreground subject segmentation (GrabCut), before/after comparison, transparent PNG download |
| **🎚️ Mood & Style Transformer** | Optical color temperature (Cold $\leftrightarrow$ Warm), dynamic chaos sliders, and geometric studio filters |
| **📈 Emotional Journey & Lore PDF** | Interactive Plotly trajectory chart of valence/arousal, session history, and ReportLab Storybook PDF export |

---

## 🎭 Where the System's Behaviour Expresses the Idea

1. **Particle Speed & Turbulence (`visuals.py`):**
   When `arousal` is low and `valence` is negative (e.g. sadness), p5.js particle speed drops to $0.35\times$ with gentle laminar drift. When `valence` and `arousal` surge (e.g. joy), speed accelerates to $1.8\times$ with radiant golden starlight turbulence.
2. **Harmonic Acoustics (`audio.py`):**
   Sadness triggers low-pass filtered pink noise simulating windowpane rain and melancholic sub-bass drones. Joy triggers radiant major harmonies (E4/B4) and uplifting TTS pacing.
3. **Cinematic Framing & Prompts (`story.py`):**
   The locked character (*Babu the Clockwork Sparrow*) is rendered in rainy indigo low-angles for melancholy, shifting seamlessly into golden-hour sunbeams with subsurface scattering when joy is embraced.
4. **Interactive Branching Momentum (`app.py` & `story.py`):**
   Story choices reflect your internal state—offering quiet, tender reflection during sorrow, and bold, soaring leaps during inspiration.
