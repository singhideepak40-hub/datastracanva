# ✨ Dastaan Canvas (दास्तान कैनवास)

> **Art that feels what you feel, and weaves a living tale scene by scene.**  
> A multimodal AI visual system whose **behaviour is the message**—where emotion dynamically dictates color palettes, fluid physics, cinematic pacing, soundscapes, and branching narrative paths.  
> Uniting the dark, high-precision canvas aesthetic of **Pokecut** with the magical, nostalgic wonder of **Toy Story**.

---

## 🌟 3-Line Concept Pitch
1. **Emotional Symbiosis:** Rather than just creating static images, Dastaan Canvas listens to your emotional coordinates (valence and arousal) via text, voice, camera, or image upload.
2. **Behavioral Expression:** System physics—p5.js particle velocity, color temperature, sound harmonics, and narrative tension—visibly and audibly transform to reflect your state of mind.
3. **Living Storybook:** Persistent heroes (like *Babu the Clockwork Sparrow*) embark on a branching cinematic journey where your choices and emotions steer each chapter.

---

## 🚀 Quick Setup & Installation

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/singhideepak40-hub/datastracanva.git
cd datastracanva
pip install -r requirements.txt
```

### 2. Configure Environment Variables / Secrets (Optional)
Gemini Multimodal API can be configured in `.streamlit/secrets.toml` or via system environment variables:
```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-google-gemini-api-key"
```
Or in your terminal:
```bash
export GEMINI_API_KEY="your-google-gemini-api-key"
```
*Note: If no API key is provided, Dastaan Canvas seamlessly runs its built-in lexical emotion engine and procedural artistic canvas fallback—zero crashes guaranteed.*

### 3. Launch Application
```bash
streamlit run app.py
```

---

## 🧭 Architecture & Modular Deliverables

| Module | Core Purpose |
| :--- | :--- |
| `app.py` | Sleek 3-column Pokecut studio layout (Input Studio, Living Canvas, Evolving Lore) |
| `emotion.py` | Multimodal Emotion Engine (Text, Consent-enforced Webcam, Voice, Vision) |
| `story.py` | Scene-by-scene branching narrative with locked characters, settings, and multi-language support (English, Hindi, Hinglish) |
| `visuals.py` | Reactive p5.js fluid canvas, Pollinations AI diffusion, procedural fallback art, Ken Burns zoom, and optical Mood Transform sliders |
| `audio.py` | Browser-native Web Speech TTS narration and real-time Web Audio API ambient soundscapes (rain, celestial chimes, ocean drone) |
| `storage.py` | SQLite Memory Wall (`dastaan_memory.db`), interactive Plotly emotion timeline, and ReportLab Storybook PDF export |
| `config/emotion_map.json` | Central customizable mapping: emotion $\rightarrow$ palette, brush texture, motion speed, camera style, and soundscape |

---

## 🎭 Where the System's Behaviour Expresses the Idea

1. **Particle Speed & Turbulence (`visuals.py`):**
   When `arousal` is low and `valence` is negative (e.g. sadness), p5.js particle speed drops to $0.35\times$ with gentle laminar drift. When `valence` and `arousal` surge (e.g. joy), speed accelerates to $1.8\times$ with radiant starlight turbulence.
2. **Harmonic Acoustics (`audio.py`):**
   Sadness triggers low-pass filtered pink noise simulating windowpane rain and melancholic sub-bass drones. Joy triggers radiant major harmonies (E4/B4) and uplifting TTS pacing.
3. **Cinematic Framing & Prompts (`story.py`):**
   The locked character (*Babu the Clockwork Sparrow*) is rendered in rainy indigo low-angles for melancholy, shifting seamlessly into golden-hour sunbeams with subsurface scattering when joy is embraced.
4. **Interactive Branching Momentum (`app.py` & `story.py`):**
   Story choices reflect your internal state—offering quiet, tender reflection during sorrow, and bold, soaring leaps during inspiration.
