"""
Audio Module for Dastaan Canvas
Handles:
- Procedural Web Audio API ambient soundscapes matched to emotion (rain, celestial chimes, calming ocean, dramatic drone)
- Client-side Web Speech API Text-to-Speech narration with language pitch/tone adjustments
- Full mute controls and autoplay compliance
"""

import streamlit.components.v1 as components

class AudioManager:
    @staticmethod
    def render_audio_controller(
        narration_text: str = "",
        emotion_name: str = "neutral",
        sound_type: str = "forest_breeze",
        language: str = "English",
        auto_narrate: bool = False
    ):
        """
        Embeds a sleek, modern audio controller with mood-matched synth ambient soundscapes
        and browser-native Web Speech TTS narration.
        [BEHAVIOUR EXPRESSION]: The fundamental sound frequencies, synth harmonics, and acoustic pacing
        shift from melancholic minor rain drones to sparkling celestial major chords.
        """
        clean_narration = (
            (narration_text or "")
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace("'", "\\'")
            .replace("\n", " ")
            .replace("\r", " ")
        )

        lang_code = "en-US"
        if language == "Hindi":
            lang_code = "hi-IN"
        elif language == "Hinglish":
            lang_code = "hi-IN"

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            .audio-panel {{
              background: rgba(22, 27, 34, 0.7);
              border: 1px solid rgba(255, 255, 255, 0.08);
              backdrop-filter: blur(12px);
              border-radius: 14px;
              padding: 10px 16px;
              display: flex;
              align-items: center;
              justify-content: space-between;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
              color: #E6EDF3;
              margin-top: 8px;
            }}
            .btn-group {{
              display: flex;
              gap: 8px;
              align-items: center;
            }}
            .ctrl-btn {{
              background: #238636;
              color: white;
              border: none;
              border-radius: 8px;
              padding: 6px 14px;
              font-size: 12px;
              font-weight: 600;
              cursor: pointer;
              transition: all 0.2s ease;
              display: flex;
              align-items: center;
              gap: 5px;
            }}
            .ctrl-btn:hover {{
              opacity: 0.9;
              transform: translateY(-1px);
            }}
            .ctrl-btn.secondary {{
              background: #30363D;
              color: #C9D1D9;
            }}
            .ctrl-btn.secondary:hover {{
              background: #484F58;
            }}
            .status-badge {{
              font-size: 11px;
              color: #8B949E;
              display: flex;
              align-items: center;
              gap: 6px;
            }}
            .status-dot {{
              width: 8px;
              height: 8px;
              border-radius: 50%;
              background: #58A6FF;
              display: inline-block;
            }}
            .status-dot.active {{
              background: #3FB950;
              box-shadow: 0 0 8px #3FB950;
            }}
          </style>
        </head>
        <body>
          <div class="audio-panel">
            <div class="btn-group">
              <button class="ctrl-btn" id="ttsBtn" onclick="toggleNarration()">
                <span id="ttsIcon">🔊</span> <span id="ttsLabel">Listen Narration</span>
              </button>
              <button class="ctrl-btn secondary" id="ambientBtn" onclick="toggleAmbient()">
                <span id="ambIcon">🎵</span> <span id="ambLabel">Ambient Soundscape</span>
              </button>
            </div>
            <div class="status-badge">
              <span class="status-dot" id="indicatorDot"></span>
              <span id="statusText">Mode: {sound_type.replace('_', ' ').capitalize()}</span>
            </div>
          </div>

          <script>
            let audioCtx = null;
            let ambientOsc1 = null;
            let ambientOsc2 = null;
            let noiseNode = null;
            let gainNode = null;
            let isAmbientPlaying = false;
            let synth = window.speechSynthesis;
            let isNarrating = false;

            const narrationText = "{clean_narration}";
            const soundType = "{sound_type}";
            const emotion = "{emotion_name}";
            const langCode = "{lang_code}";

            function initAudio() {{
              if (!audioCtx) {{
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
              }}
              if (audioCtx.state === 'suspended') {{
                audioCtx.resume();
              }}
            }}

            function toggleAmbient() {{
              initAudio();
              const btn = document.getElementById('ambientBtn');
              const dot = document.getElementById('indicatorDot');
              const status = document.getElementById('statusText');

              if (isAmbientPlaying) {{
                stopAmbient();
                btn.style.background = '#30363D';
                dot.classList.remove('active');
                status.innerText = 'Ambient Paused';
              }} else {{
                playAmbient();
                btn.style.background = '#1F6FEB';
                dot.classList.add('active');
                status.innerText = 'Ambient: ' + soundType.replace('_', ' ');
              }}
            }}

            function playAmbient() {{
              stopAmbient();
              gainNode = audioCtx.createGain();
              gainNode.gain.setValueAtTime(0.08, audioCtx.currentTime);
              gainNode.connect(audioCtx.destination);

              if (soundType === 'gentle_rain') {{
                // Rain / melancholic pink noise generator
                const bufferSize = audioCtx.sampleRate * 2;
                const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
                const data = buffer.getChannelData(0);
                let b0 = 0, b1 = 0, b2 = 0;
                for (let i = 0; i < bufferSize; i++) {{
                  let white = Math.random() * 2 - 1;
                  b0 = 0.99886 * b0 + white * 0.0555179;
                  b1 = 0.99332 * b1 + white * 0.0750759;
                  b2 = 0.96900 * b2 + white * 0.1538520;
                  data[i] = (b0 + b1 + b2) * 0.11;
                }}
                noiseNode = audioCtx.createBufferSource();
                noiseNode.buffer = buffer;
                noiseNode.loop = true;

                const filter = audioCtx.createBiquadFilter();
                filter.type = 'lowpass';
                filter.frequency.value = 850;
                noiseNode.connect(filter);
                filter.connect(gainNode);
                noiseNode.start();

              }} else if (soundType === 'warm_celestial') {{
                // Sparkly harmonic chord (A440 + E660 major harmony)
                ambientOsc1 = audioCtx.createOscillator();
                ambientOsc2 = audioCtx.createOscillator();
                ambientOsc1.type = 'sine';
                ambientOsc2.type = 'triangle';
                ambientOsc1.frequency.setValueAtTime(329.63, audioCtx.currentTime); // E4
                ambientOsc2.frequency.setValueAtTime(493.88, audioCtx.currentTime); // B4

                ambientOsc1.connect(gainNode);
                ambientOsc2.connect(gainNode);
                ambientOsc1.start();
                ambientOsc2.start();

              }} else if (soundType === 'dramatic_pulse') {{
                // Deep ominous tension pulse
                ambientOsc1 = audioCtx.createOscillator();
                ambientOsc1.type = 'sawtooth';
                ambientOsc1.frequency.setValueAtTime(65.4, audioCtx.currentTime); // C2 drone
                const filter = audioCtx.createBiquadFilter();
                filter.type = 'lowpass';
                filter.frequency.value = 350;
                ambientOsc1.connect(filter);
                filter.connect(gainNode);
                ambientOsc1.start();

              }} else {{
                // Gentle breeze sine drift
                ambientOsc1 = audioCtx.createOscillator();
                ambientOsc1.type = 'sine';
                ambientOsc1.frequency.setValueAtTime(220.0, audioCtx.currentTime); // A3
                ambientOsc1.connect(gainNode);
                ambientOsc1.start();
              }}
              isAmbientPlaying = true;
            }}

            function stopAmbient() {{
              if (ambientOsc1) {{ try {{ ambientOsc1.stop(); ambientOsc1.disconnect(); }} catch(e){{}} ambientOsc1 = null; }}
              if (ambientOsc2) {{ try {{ ambientOsc2.stop(); ambientOsc2.disconnect(); }} catch(e){{}} ambientOsc2 = null; }}
              if (noiseNode) {{ try {{ noiseNode.stop(); noiseNode.disconnect(); }} catch(e){{}} noiseNode = null; }}
              isAmbientPlaying = false;
            }}

            function toggleNarration() {{
              if (!synth) return;
              const btn = document.getElementById('ttsBtn');
              const label = document.getElementById('ttsLabel');
              const icon = document.getElementById('ttsIcon');

              if (synth.speaking && !synth.paused) {{
                synth.cancel();
                isNarrating = false;
                label.innerText = 'Listen Narration';
                icon.innerText = '🔊';
                btn.style.background = '#238636';
              }} else {{
                synth.cancel();
                if (!narrationText || narrationText.trim() === '') return;
                const utter = new SpeechSynthesisUtterance(narrationText);
                utter.lang = langCode;
                utter.rate = emotion === 'sadness' ? 0.85 : 1.05;
                utter.pitch = emotion === 'sadness' ? 0.9 : 1.1;

                utter.onend = () => {{
                  isNarrating = false;
                  label.innerText = 'Listen Narration';
                  icon.innerText = '🔊';
                  btn.style.background = '#238636';
                }};

                synth.speak(utter);
                isNarrating = true;
                label.innerText = 'Stop Reading';
                icon.innerText = '⏹️';
                btn.style.background = '#DA3633';
              }}
            }}
          </script>
        </body>
        </html>
        """
        components.html(html_code, height=64, scrolling=False)
