"""
Storage and Analytics Module for Dastaan Canvas
Handles:
- Emotion Timeline Interactive Plotly Graph (Valence & Arousal journey)
- SQLite Session Archival & Memory Wall / Gallery
- Storybook PDF Export via ReportLab
- Poster Card / HTML Export
"""

import os
import io
import json
import sqlite3
import datetime
import html
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from PIL import Image

DB_PATH = os.path.join(os.path.dirname(__file__), "dastaan_memory.db")

class MemoryStore:
    @staticmethod
    def init_db():
        """Initializes SQLite database for session memory and replay wall."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_title TEXT,
                timestamp TEXT,
                character_name TEXT,
                setting_name TEXT,
                language TEXT,
                scenes_json TEXT,
                emotion_summary TEXT
            )
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def save_session(session_title: str, character_name: str, setting_name: str, language: str, scenes: list):
        """Saves completed or ongoing session into SQLite."""
        MemoryStore.init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        emotions = [s.get("emotion", "neutral") for s in scenes]
        summary = f"{len(scenes)} scenes | Moods: {', '.join(set(emotions))}"

        cursor.execute("""
            INSERT INTO sessions (session_title, timestamp, character_name, setting_name, language, scenes_json, emotion_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_title, ts, character_name, setting_name, language, json.dumps(scenes), summary))
        conn.commit()
        conn.close()

    @staticmethod
    def list_saved_sessions():
        """Lists all recorded creative sessions."""
        MemoryStore.init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, session_title, timestamp, character_name, emotion_summary, scenes_json FROM sessions ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows


# =========================================================================
# FEATURE 15: Emotion Timeline Graph (Plotly Valence / Arousal Trajectory)
# =========================================================================
class TimelineVisualizer:
    @staticmethod
    def create_emotion_timeline_chart(scenes: list):
        """
        Builds a dark-themed cinematic Plotly chart showing the emotional arc across scenes.
        - X-axis: Scene Number
        - Y-axis: Valence (-1 to +1) & Arousal (0 to 1)
        """
        if not scenes:
            fig = go.Figure()
            fig.update_layout(
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                annotations=[dict(text="Awaiting first scene to chart emotional journey...", showarrow=False, font=dict(color="#8B949E", size=13))]
            )
            return fig

        scene_nums = [f"Scene {s.get('scene_number', i+1)}" for i, s in enumerate(scenes)]
        valences = [s.get("valence", 0.0) for s in scenes]
        arousals = [s.get("arousal", 0.4) for s in scenes]
        emotions = [s.get("emotion", "neutral").upper() for s in scenes]

        fig = go.Figure()

        # Valence Line (Emotional Sentiment: Joy vs Sorrow)
        fig.add_trace(go.Scatter(
            x=scene_nums,
            y=valences,
            mode='lines+markers+text',
            name='Valence (Mood Tone)',
            line=dict(color='#58A6FF', width=3, shape='spline'),
            marker=dict(size=10, color='#58A6FF', symbol='circle'),
            text=emotions,
            textposition='top center',
            textfont=dict(color='#E6EDF3', size=10)
        ))

        # Arousal Line (Kinetic Energy & Intensity)
        fig.add_trace(go.Scatter(
            x=scene_nums,
            y=arousals,
            mode='lines+markers',
            name='Arousal (Energy Level)',
            line=dict(color='#F778BA', width=2.5, dash='dot', shape='spline'),
            marker=dict(size=8, color='#F778BA', symbol='diamond')
        ))

        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            margin=dict(l=30, r=30, t=35, b=30),
            height=260,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#C9D1D9", size=10)
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
                tickfont=dict(color="#8B949E", size=10)
            ),
            yaxis=dict(
                range=[-1.1, 1.2],
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
                tickfont=dict(color="#8B949E", size=10),
                zeroline=True,
                zerolinecolor="rgba(255, 255, 255, 0.2)"
            )
        )
        return fig


# =========================================================================
# FEATURE 19: Export Engine (Storybook PDF & Poster Card)
# =========================================================================
class ExportEngine:
    @staticmethod
    def generate_storybook_pdf(session_title: str, character: dict, setting: dict, scenes: list) -> bytes:
        """Compiles story scenes, titles, and narratives into a printable Storybook PDF."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1A202C"),
            alignment=1,
            spaceAfter=12
        )
        sub_style = ParagraphStyle(
            name="SubStyle",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#4A5568"),
            alignment=1,
            spaceAfter=20
        )
        scene_title_style = ParagraphStyle(
            name="SceneTitle",
            parent=styles["Heading2"],
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=14,
            spaceAfter=6
        )
        narrative_style = ParagraphStyle(
            name="NarrativeStyle",
            parent=styles["BodyText"],
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#2D3748"),
            spaceAfter=14
        )

        safe_title = html.escape(str(session_title or "Dastaan Storybook"))
        safe_char = html.escape(str(character.get('name', 'Hero')))
        safe_set = html.escape(str(setting.get('name', 'Enchanted Realm')))

        elements = []
        elements.append(Paragraph(f"✨ Dastaan Canvas: {safe_title}", title_style))
        elements.append(Paragraph(f"Hero: {safe_char} | Setting: {safe_set}", sub_style))
        elements.append(Spacer(1, 10))

        for s in (scenes or []):
            stitle = html.escape(str(s.get('title', 'Scene')))
            snarr = html.escape(str(s.get('narrative', '')))
            elements.append(Paragraph(f"◈ {stitle}", scene_title_style))
            elements.append(Paragraph(snarr, narrative_style))
            elements.append(Spacer(1, 8))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def generate_html_poster(scene: dict, character: dict) -> str:
        """Generates a shareable HTML card snapshot with Pokecut aesthetic styling."""
        return f"""
        <div style="
          background: linear-gradient(135deg, #131722 0%, #1A2234 100%);
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-radius: 18px;
          padding: 24px;
          color: white;
          font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          max-width: 600px;
          margin: auto;
          box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        ">
          <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #58A6FF; margin-bottom: 8px;">
            ✨ DASTAAN CANVAS COLLECTIBLE
          </div>
          <h2 style="margin: 0 0 10px 0; font-size: 20px; color: #FFF;">{scene.get('title', 'Story Scene')}</h2>
          <p style="font-size: 13px; line-height: 1.6; color: #C9D1D9; margin-bottom: 14px;">
            "{scene.get('narrative', '')}"
          </p>
          <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px; font-size: 11px; color: #8B949E;">
            <span>Hero: {character.get('name')}</span>
            <span>Mood: {scene.get('emotion', 'wonder').upper()}</span>
          </div>
        </div>
        """
