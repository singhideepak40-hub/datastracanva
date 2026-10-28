import random

class StoryGenerator:
    @staticmethod
    def generate_narrative(mood: str, features: dict) -> dict:
        brightness = features.get("brightness", 128)
        saturation = features.get("saturation", 100)
        
        # Determine Context Title & Magical Lore
        if saturation > 120:
            title = "The Realm of Luminous Fables"
            thematic_arc = "A vibrant awakening of color, where every hue tells an ancient secret of light."
            story = (
                "In a dimension where pigments breathe and shadows dance, this snapshot captures the precise heartbeat "
                "of an eternal morning. Sunlight cascades across the canvas like liquid gold, whispering tales of forgotten "
                "kingdoms where the sky is ever-blazing and wonder knows no horizon. Every pixel radiates an unstoppable pulse "
                "of life, inviting travelers to step beyond the veil into pure enchantment."
            )
        elif brightness < 80:
            title = "Chronicles of the Obsidian Veil"
            thematic_arc = "A quiet nocturnal mystery where whispers linger in the velvet dark."
            story = (
                "Beneath a canopy of silent stars and silver moonlight lies a world veiled in deep obsidian shadows. "
                "Here, time slows to a gentle murmur. The darkness is not empty; it is alive with untold secrets, "
                "ancient nocturnal echoes, and the quiet magic of souls who thrive when the rest of the realm is fast asleep."
            )
        else:
            title = "The Alchemist's Timeless Echo"
            thematic_arc = "A delicate equilibrium between light and memory, preserved in stardust."
            story = (
                "A timeless fragment caught between the amber glow of yesterday and the starry promise of tomorrow. "
                "The air shimmers with nostalgic warmth, as though the very atmosphere holds the memories of wanderers "
                "who once walked this enchanted path. A true masterpiece of visual poetry."
            )
            
        return {
            "title": title,
            "thematic_arc": thematic_arc,
            "story": story,
            "mood_summary": f"Detected Mood: {mood} (Brightness: {brightness:.1f}, Saturation: {saturation:.1f})"
        }