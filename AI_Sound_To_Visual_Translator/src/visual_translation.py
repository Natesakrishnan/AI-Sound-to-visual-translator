"""
Module 4: Sound-to-Visual Translation Engine
Converts physical and spectral audio characteristics + CNN predicted classes
into deterministic, mathematically grounded, aesthetic visual geometries.

Mathematical Translation Mappings:
  - RMS Energy          -> Visual Radius / Global Scale / Glow Intensity
  - Spectral Centroid   -> Angular Rotation / Color Spectrum Distribution
  - Zero Crossing Rate  -> Vertex Count / Edge Complexity / Micro-Frequencies
  - Dominant Frequency  -> Harmonic Spiral Tightness / Waveform Deformation
  - Chroma Pitch Class  -> Chromatic Palette & Primary Color Hue
  - CNN Predicted Class -> Base Sacred / Generative Geometry Archetype (50 ESC-50 classes)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_extraction import extract_all_visual_features


# Map each of the 50 ESC-50 classes into one of 5 foundational geometric archetypes
CLASS_ARCHETYPES = {
    # 1. Animals -> Organic Radial Petals & Orbital Rings
    "dog": "animal_orbital", "rooster": "animal_orbital", "pig": "animal_orbital",
    "cow": "animal_orbital", "frog": "animal_orbital", "cat": "animal_orbital",
    "hen": "animal_orbital", "insects": "animal_orbital", "sheep": "animal_orbital",
    "crow": "animal_orbital",

    # 2. Natural Soundscapes & Water -> Fluid Wave Cascades & Flow Ribbons
    "rain": "fluid_wave", "sea_waves": "fluid_wave", "crackling_fire": "fluid_wave",
    "crickets": "fluid_wave", "chirping_birds": "fluid_wave", "water_drops": "fluid_wave",
    "wind": "fluid_wave", "pouring_water": "fluid_wave", "toilet_flush": "fluid_wave",
    "thunderstorm": "fluid_wave",

    # 3. Human Non-Speech -> Harmonic Acoustic Nodes & Breathing Auras
    "crying_baby": "harmonic_node", "sneezing": "harmonic_node", "clapping": "harmonic_node",
    "breathing": "harmonic_node", "coughing": "harmonic_node", "footsteps": "harmonic_node",
    "laughing": "harmonic_node", "brushing_teeth": "harmonic_node", "snoring": "harmonic_node",
    "drinking_sipping": "harmonic_node",

    # 4. Interior / Domestic -> Structured Geometric Tessellations & Lattice Nodes
    "door_wood_creaks": "lattice_grid", "mouse_click": "lattice_grid", "keyboard_typing": "lattice_grid",
    "door_wood_knock": "lattice_grid", "can_opening": "lattice_grid", "washing_machine": "lattice_grid",
    "vacuum_cleaner": "lattice_grid", "clock_alarm": "lattice_grid", "clock_tick": "lattice_grid",
    "glass_breaking": "lattice_grid",

    # 5. Exterior / Urban -> Crystalline Spirals & High-Energy Angular Polygons
    "helicopter": "angular_burst", "chainsaw": "angular_burst", "siren": "angular_burst",
    "car_horn": "angular_burst", "engine": "angular_burst", "train": "angular_burst",
    "church_bells": "angular_burst", "airplane": "angular_burst", "fireworks": "angular_burst",
    "hand_saw": "angular_burst"
}


# Visual Modes supported by the application
VISUAL_MODES = [
    "Class-Specific Generative Geometry",
    "Sound Mandala / Energy Aura",
    "Frequency Harmonic Spiral",
    "Waveform Geometric Flower",
    "Acoustic Particle Constellation"
]


class SoundVisualTranslator:
    """
    Translates extracted audio features and CNN classifications into deterministic visual shapes.
    """
    def __init__(self, output_dir: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "outputs" / "generated_visuals"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_visual_parameters(
        self,
        features: Dict[str, Any],
        predicted_class: str = "unknown",
        confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Execute deterministic mathematical mapping from raw physical features to visual rendering parameters.
        """
        rms_norm = features["rms_norm"]              # [0.05, 1.0]
        centroid_norm = features["centroid_norm"]    # [0.02, 1.0]
        zcr_norm = features["zcr_norm"]              # [0.02, 1.0]
        dom_freq_norm = features["dom_freq_norm"]    # [0.01, 1.0]
        pitch_idx = features["dominant_pitch_idx"]   # 0 to 11

        # 1. Base Radius: Driven by RMS Energy (Scale factor from 0.8 to 2.4)
        base_radius = 1.0 + 1.4 * rms_norm

        # 2. Global Rotation: Driven by Spectral Centroid (0 to 2*pi)
        rotation_angle = centroid_norm * 2.0 * np.pi

        # 3. Geometric Complexity / Number of Vertices / Lobes: Driven by ZCR
        vertices = int(4 + np.round(28.0 * zcr_norm))  # 4 to 32 vertices

        # 4. Harmonic Layer Count / Dispersion: Driven by Dominant Frequency
        harmonic_layers = int(2 + np.round(6.0 * dom_freq_norm))  # 2 to 8 layers

        # 5. Primary Color Hue: Driven by dominant musical pitch note (0 to 11 -> 0.0 to 1.0)
        hue_primary = (pitch_idx * (360.0 / 12.0)) / 360.0

        # 6. Secondary / Complementary Color Hue: Shifted by Spectral Centroid
        hue_secondary = (hue_primary + 0.35 + 0.3 * centroid_norm) % 1.0

        # 7. Deformation Amplitude: Waveform roughness driven by RMS max and ZCR
        deformation_scale = 0.15 + 0.45 * features["rms_max_norm"]

        # 8. Identify Generative Archetype
        archetype = CLASS_ARCHETYPES.get(predicted_class.lower(), "animal_orbital")

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "archetype": archetype,
            "base_radius": round(base_radius, 3),
            "rotation_angle_rad": round(rotation_angle, 3),
            "rotation_angle_deg": round(np.degrees(rotation_angle), 1),
            "vertices": vertices,
            "harmonic_layers": harmonic_layers,
            "hue_primary": round(hue_primary, 3),
            "hue_secondary": round(hue_secondary, 3),
            "deformation_scale": round(deformation_scale, 3),
            "raw_rms": features["raw_rms_mean"],
            "raw_centroid": features["raw_centroid_hz"],
            "raw_zcr": features["raw_zcr_mean"],
            "raw_dom_freq": features["raw_dominant_freq_hz"],
            "pitch_note": features["dominant_pitch"]
        }

    def render_visual(
        self,
        y: np.ndarray,
        sr: int = 22050,
        predicted_class: str = "sound",
        confidence: float = 0.0,
        visual_mode: str = "Class-Specific Generative Geometry",
        figsize: Tuple[int, int] = (8, 8),
        dpi: int = 150
    ) -> Tuple[Figure, Dict[str, Any]]:
        """
        Generate a Matplotlib Figure rendering the sound-to-visual translation.
        """
        features = extract_all_visual_features(y, sr=sr)
        params = self.compute_visual_parameters(features, predicted_class, confidence)

        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#0d1117")
        ax = fig.add_subplot(111, facecolor="#0d1117")
        ax.set_aspect("equal")
        ax.axis("off")

        # Color generation from hue parameters
        color1 = mcolors.hsv_to_rgb([params["hue_primary"], 0.85, 0.95])
        color2 = mcolors.hsv_to_rgb([params["hue_secondary"], 0.90, 0.90])
        color_accent = mcolors.hsv_to_rgb([(params["hue_primary"] + 0.5) % 1.0, 0.75, 1.0])

        # Dispatch rendering mode
        if visual_mode == "Sound Mandala / Energy Aura":
            self._render_mandala(ax, params, features, color1, color2, color_accent)
        elif visual_mode == "Frequency Harmonic Spiral":
            self._render_spiral(ax, params, features, color1, color2, color_accent)
        elif visual_mode == "Waveform Geometric Flower":
            self._render_flower(ax, params, features, color1, color2, color_accent)
        elif visual_mode == "Acoustic Particle Constellation":
            self._render_constellation(ax, params, features, color1, color2, color_accent)
        else:
            # Class-Specific Generative Geometry (Adaptive based on ESC-50 class archetype)
            arch = params["archetype"]
            if arch == "fluid_wave":
                self._render_fluid_waves(ax, params, features, color1, color2, color_accent)
            elif arch == "harmonic_node":
                self._render_harmonic_nodes(ax, params, features, color1, color2, color_accent)
            elif arch == "lattice_grid":
                self._render_lattice(ax, params, features, color1, color2, color_accent)
            elif arch == "angular_burst":
                self._render_angular_burst(ax, params, features, color1, color2, color_accent)
            else:
                self._render_animal_orbital(ax, params, features, color1, color2, color_accent)

        # Telemetry & Title Label on Figure
        title_text = f"AI Sound-to-Visual Translation: [{predicted_class.upper()}]"
        conf_text = f"Confidence: {confidence:.1f}% | Pitch: {params['pitch_note']} | Centroid: {params['raw_centroid']} Hz"
        ax.text(0.5, 0.96, title_text, transform=fig.transFigure,
                ha="center", va="top", color="#58a6ff", fontsize=12, weight="bold", fontfamily="sans-serif")
        ax.text(0.5, 0.92, conf_text, transform=fig.transFigure,
                ha="center", va="top", color="#8b949e", fontsize=9, fontfamily="sans-serif")

        plt.tight_layout(pad=1.5)
        return fig, params

    def save_visual_png(
        self,
        fig: Figure,
        predicted_class: str = "sound",
        custom_filename: Optional[str] = None
    ) -> str:
        """
        Save the generated Matplotlib visual to a PNG file on disk.
        Supports both absolute target file paths and relative custom filenames.
        """
        if custom_filename:
            target_p = Path(custom_filename)
            if target_p.is_absolute():
                save_path = target_p if target_p.suffix.lower() == ".png" else target_p.with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                safe_custom = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in custom_filename)
                fname = safe_custom if safe_custom.endswith(".png") else f"{safe_custom}.png"
                save_path = self.output_dir / fname
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_class = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in predicted_class.lower())
            fname = f"visual_{safe_class}_{timestamp}.png"
            save_path = self.output_dir / fname

        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"[VisualTranslator] Saved generated visual to: {save_path}")
        return str(save_path)

    # -------------------------------------------------------------
    # GENERATIVE RENDER ENGINES (DETERMINISTIC MATHEMATICAL SHAPES)
    # -------------------------------------------------------------

    def _render_mandala(self, ax, p, feats, c1, c2, c_acc):
        """Sound Mandala / Energy Aura: Nested modulated geometric rings."""
        theta = np.linspace(0, 2 * np.pi, 600)
        n_layers = p["harmonic_layers"]
        k_lobes = p["vertices"]
        rot = p["rotation_angle_rad"]

        for i in range(1, n_layers + 1):
            r_scale = (i / n_layers) * p["base_radius"]
            # Modulated radius with frequency harmonics
            r = r_scale * (1.0 + p["deformation_scale"] * np.sin(k_lobes * (theta + rot * i) + i * np.pi / 4))
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            alpha = 0.35 + 0.55 * (i / n_layers)
            blend = i / n_layers
            color = (1 - blend) * c1 + blend * c2
            ax.plot(x, y, color=color, linewidth=1.8 + blend, alpha=alpha)

            # Draw radial connective lines at symmetry nodes
            if i == n_layers:
                node_angles = np.linspace(0, 2 * np.pi, k_lobes, endpoint=False) + rot
                node_r = p["base_radius"] * (1.0 + p["deformation_scale"])
                for na in node_angles:
                    ax.plot([0, node_r * np.cos(na)], [0, node_r * np.sin(na)],
                            color=c_acc, linewidth=1.0, alpha=0.4, linestyle=":")
                    ax.scatter([node_r * np.cos(na)], [node_r * np.sin(na)],
                               color=c_acc, s=35, alpha=0.8, zorder=5)

        # Glowing central core
        core_circle = plt.Circle((0, 0), p["base_radius"] * 0.15, color=c1, alpha=0.6, zorder=4)
        ax.add_patch(core_circle)
        ax.scatter([0], [0], color="#ffffff", s=60, zorder=6)
        ax.set_xlim(-p["base_radius"] * 1.6, p["base_radius"] * 1.6)
        ax.set_ylim(-p["base_radius"] * 1.6, p["base_radius"] * 1.6)

    def _render_spiral(self, ax, p, feats, c1, c2, c_acc):
        """Frequency Harmonic Spiral: Multi-arm logarithmic spirals."""
        n_turns = 3.5 + 4.0 * feats["dom_freq_norm"]
        theta = np.linspace(0, n_turns * 2 * np.pi, 1000)
        k_arms = max(2, p["vertices"] // 4)
        rot = p["rotation_angle_rad"]

        for arm in range(k_arms):
            arm_offset = (arm * 2 * np.pi) / k_arms + rot
            # Archimedean/Logarithmic hybrid spiral modulated by RMS
            r = (0.1 + p["base_radius"] * (theta / (n_turns * 2 * np.pi))**0.9)
            # Amplitude waviness
            r_mod = r * (1.0 + 0.18 * np.sin(8 * theta + arm))
            x = r_mod * np.cos(theta + arm_offset)
            y = r_mod * np.sin(theta + arm_offset)

            blend = arm / k_arms
            color = (1 - blend) * c1 + blend * c2
            ax.plot(x, y, color=color, linewidth=2.2, alpha=0.75)

            # Node markers along the spiral
            step = len(theta) // 12
            ax.scatter(x[::step], y[::step], color=c_acc, s=25, alpha=0.6, zorder=4)

        ax.set_xlim(-p["base_radius"] * 1.5, p["base_radius"] * 1.5)
        ax.set_ylim(-p["base_radius"] * 1.5, p["base_radius"] * 1.5)

    def _render_flower(self, ax, p, feats, c1, c2, c_acc):
        """Waveform Geometric Flower: Rose/Rhodonea parametric curves."""
        theta = np.linspace(0, 2 * np.pi, 800)
        k_petals = max(3, p["vertices"] // 2)
        rot = p["rotation_angle_rad"]

        for layer in range(1, p["harmonic_layers"] + 1):
            r_base = (layer / p["harmonic_layers"]) * p["base_radius"]
            # Rhodonea curve r = cos(k * theta)
            r = r_base * (0.4 + 0.6 * np.abs(np.cos(k_petals * (theta + rot * layer))))
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            blend = layer / p["harmonic_layers"]
            color = (1 - blend) * c1 + blend * c2
            ax.fill(x, y, color=color, alpha=0.20, zorder=layer)
            ax.plot(x, y, color=color, linewidth=2.0, alpha=0.85, zorder=layer + 1)

        ax.set_xlim(-p["base_radius"] * 1.4, p["base_radius"] * 1.4)
        ax.set_ylim(-p["base_radius"] * 1.4, p["base_radius"] * 1.4)

    def _render_constellation(self, ax, p, feats, c1, c2, c_acc):
        """Acoustic Particle Constellation: Orbital particle network."""
        np.random.seed(42)  # Deterministic base placement
        n_particles = int(35 + 80 * feats["zcr_norm"])
        angles = np.linspace(0, 2 * np.pi, n_particles, endpoint=False) + p["rotation_angle_rad"]
        radii = p["base_radius"] * (0.3 + 0.8 * np.random.rand(n_particles) * (1.0 + feats["rms_norm"]))

        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)

        # Draw connecting web for nearby particles
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                dist = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                if dist < p["base_radius"] * 0.45:
                    alpha = float(np.clip(1.0 - dist / (p["base_radius"] * 0.45), 0.1, 0.6))
                    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=c1, alpha=alpha * 0.5, linewidth=0.8)

        # Scatter points
        sizes = 20 + 70 * np.random.rand(n_particles) * p["base_radius"]
        ax.scatter(xs, ys, color=c_acc, s=sizes, alpha=0.85, zorder=5, edgecolors=c2, linewidths=1.0)
        ax.scatter([0], [0], color="#ffffff", s=100, alpha=0.9, zorder=6)
        ax.set_xlim(-p["base_radius"] * 1.5, p["base_radius"] * 1.5)
        ax.set_ylim(-p["base_radius"] * 1.5, p["base_radius"] * 1.5)

    def _render_animal_orbital(self, ax, p, feats, c1, c2, c_acc):
        """Archetype 1: Animals -> Bio-harmonic concentric orbital aura."""
        theta = np.linspace(0, 2 * np.pi, 600)
        rot = p["rotation_angle_rad"]
        n_rings = max(3, p["harmonic_layers"])
        petals = max(4, p["vertices"] // 3)

        for i in range(1, n_rings + 1):
            r_scale = (i / n_rings) * p["base_radius"]
            modulation = p["deformation_scale"] * np.sin(petals * theta + rot * i)
            r = r_scale * (1.0 + modulation)
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            blend = i / n_rings
            col = (1 - blend) * c1 + blend * c2
            ax.plot(x, y, color=col, linewidth=2.0 + blend, alpha=0.7)
            if i % 2 == 0:
                ax.fill(x, y, color=col, alpha=0.15)

        # Orbiting satellite nodes
        sat_angles = np.linspace(0, 2 * np.pi, petals, endpoint=False) + rot
        sat_r = p["base_radius"] * 1.15
        ax.scatter(sat_r * np.cos(sat_angles), sat_r * np.sin(sat_angles),
                   color=c_acc, s=45, alpha=0.9, zorder=5)

        ax.set_xlim(-p["base_radius"] * 1.5, p["base_radius"] * 1.5)
        ax.set_ylim(-p["base_radius"] * 1.5, p["base_radius"] * 1.5)

    def _render_fluid_waves(self, ax, p, feats, c1, c2, c_acc):
        """Archetype 2: Nature / Water -> Flowing fluid streamline cascades."""
        x = np.linspace(-p["base_radius"] * 1.3, p["base_radius"] * 1.3, 500)
        n_waves = max(4, p["harmonic_layers"] + 2)

        for i in range(n_waves):
            y_base = (i - n_waves / 2) * (p["base_radius"] * 2.2 / n_waves)
            freq = 1.5 + 3.0 * feats["dom_freq_norm"] + i * 0.3
            phase = p["rotation_angle_rad"] + i * 0.5
            amp = p["base_radius"] * 0.22 * (1.0 + feats["rms_norm"])
            y_wave = y_base + amp * np.sin(freq * x + phase) * np.cos(0.4 * x)

            blend = i / n_waves
            col = (1 - blend) * c1 + blend * c2
            ax.plot(x, y_wave, color=col, linewidth=2.2, alpha=0.75)
            # Droplet particles
            idx_pts = np.linspace(20, len(x) - 20, 6, dtype=int)
            ax.scatter(x[idx_pts], y_wave[idx_pts], color=c_acc, s=25, alpha=0.7)

        ax.set_xlim(-p["base_radius"] * 1.4, p["base_radius"] * 1.4)
        ax.set_ylim(-p["base_radius"] * 1.4, p["base_radius"] * 1.4)

    def _render_harmonic_nodes(self, ax, p, feats, c1, c2, c_acc):
        """Archetype 3: Human Non-Speech -> Vocal acoustic standing wave nodes."""
        theta = np.linspace(0, 2 * np.pi, 500)
        rot = p["rotation_angle_rad"]
        n_nodes = max(3, p["vertices"] // 4)

        for i in range(1, 5):
            r = p["base_radius"] * (i / 4.0) * (1.0 + 0.25 * np.cos(n_nodes * theta + rot))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            blend = i / 4.0
            col = (1 - blend) * c1 + blend * c2
            ax.plot(x, y, color=col, linewidth=2.5, alpha=0.8)
            ax.fill(x, y, color=col, alpha=0.12)

        ax.set_xlim(-p["base_radius"] * 1.4, p["base_radius"] * 1.4)
        ax.set_ylim(-p["base_radius"] * 1.4, p["base_radius"] * 1.4)

    def _render_lattice(self, ax, p, feats, c1, c2, c_acc):
        """Archetype 4: Interior / Domestic -> Mechanical geometric lattice."""
        grid_n = max(3, int(3 + 4 * feats["zcr_norm"]))
        extent = p["base_radius"] * 1.1
        coords = np.linspace(-extent, extent, grid_n)

        rot = p["rotation_angle_rad"]
        c_rot, s_rot = np.cos(rot), np.sin(rot)

        for cx in coords:
            for cy in coords:
                # Rotate point
                rx = cx * c_rot - cy * s_rot
                ry = cx * s_rot + cy * c_rot
                dist_center = np.hypot(rx, ry)
                if dist_center <= extent:
                    s_box = extent / (grid_n * 1.6) * (1.0 + 0.5 * feats["rms_norm"])
                    rect_x = [rx - s_box, rx + s_box, rx + s_box, rx - s_box, rx - s_box]
                    rect_y = [ry - s_box, ry - s_box, ry + s_box, ry + s_box, ry - s_box]
                    ax.plot(rect_x, rect_y, color=c1, linewidth=1.5, alpha=0.75)
                    ax.scatter([rx], [ry], color=c_acc, s=20, alpha=0.8)

        ax.set_xlim(-p["base_radius"] * 1.4, p["base_radius"] * 1.4)
        ax.set_ylim(-p["base_radius"] * 1.4, p["base_radius"] * 1.4)

    def _render_angular_burst(self, ax, p, feats, c1, c2, c_acc):
        """Archetype 5: Exterior / Urban -> High-energy angular polygonal burst."""
        n_rays = max(6, p["vertices"])
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False) + p["rotation_angle_rad"]
        rot = p["rotation_angle_rad"]

        # Inner and outer polygon
        r_inner = p["base_radius"] * 0.35
        r_outer = p["base_radius"] * (1.1 + 0.4 * feats["rms_norm"])

        poly_x = []
        poly_y = []
        for i, a in enumerate(angles):
            r = r_outer if i % 2 == 0 else r_inner
            poly_x.append(r * np.cos(a))
            poly_y.append(r * np.sin(a))
        poly_x.append(poly_x[0])
        poly_y.append(poly_y[0])

        ax.fill(poly_x, poly_y, color=c2, alpha=0.25)
        ax.plot(poly_x, poly_y, color=c2, linewidth=2.5, alpha=0.9)

        # Radiating burst spikes
        for a in angles[::2]:
            ax.plot([0, r_outer * 1.25 * np.cos(a + rot)],
                    [0, r_outer * 1.25 * np.sin(a + rot)],
                    color=c_acc, linewidth=1.8, alpha=0.6, linestyle="--")
            ax.scatter([r_outer * 1.25 * np.cos(a + rot)],
                       [r_outer * 1.25 * np.sin(a + rot)],
                       color=c_acc, s=40, zorder=5)

        ax.set_xlim(-p["base_radius"] * 1.5, p["base_radius"] * 1.5)
        ax.set_ylim(-p["base_radius"] * 1.5, p["base_radius"] * 1.5)


if __name__ == "__main__":
    from audio_preprocessing import load_audio
    sample_wav = r"K:/AI project/Datasets/audio/1-100032-A-0.wav"
    if os.path.exists(sample_wav):
        y_data, sr_val = load_audio(sample_wav)
        translator = SoundVisualTranslator()
        for mode in VISUAL_MODES:
            fig, par = translator.render_visual(y_data, sr=sr_val, predicted_class="dog", confidence=96.5, visual_mode=mode)
            saved = translator.save_visual_png(fig, predicted_class="dog", custom_filename=f"test_{mode.replace(' ', '_')}")
            plt.close(fig)
            print("Rendered & Saved mode:", mode, "to", saved)
