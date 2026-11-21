"""
Desktop GUI Application for AI-Based Sound-to-Visual Translation.
Built with Python Tkinter and embedded Matplotlib canvases.
Features 5 comprehensive tabs:
  1. Home (System Overview, Hardware Monitor, Model Status, Quick Actions)
  2. Audio Analysis (Waveform, FFT, MFCC, Mel Spectrogram, Stats & Player)
  3. AI Prediction (Top-1 Class, Confidence, Top-5 Rankings, Latency)
  4. Visual Translation (Real-time Mathematical Geometry Canvas & PNG Export)
  5. Model Performance (Test Metrics, Confusion Matrix, Loss/Acc Curves, Retrain)
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_preprocessing import load_audio, get_audio_metadata, AudioPlayer, TARGET_SAMPLE_RATE
from feature_extraction import (
    compute_fft,
    compute_mel_spectrogram,
    compute_mfcc,
    compute_spectral_centroid,
    compute_zero_crossing_rate,
    compute_rms_energy,
    compute_chroma,
    extract_all_visual_features
)
from prediction import AudioPredictor, ESC10_CLASSES
from visual_translation import SoundVisualTranslator, VISUAL_MODES
from hardware_info import get_system_hardware_info, get_hardware_summary_text
from train_model import train_audio_classifier


class SoundVisualizerApp(tk.Tk):
    """
    Main Desktop GUI Application Window.
    """
    # Modern Dark Theme Palette
    COLOR_BG = "#0d1117"
    COLOR_CARD = "#161b22"
    COLOR_CARD_BORDER = "#30363d"
    COLOR_TEXT = "#c9d1d9"
    COLOR_TEXT_BRIGHT = "#f0f6fc"
    COLOR_TEXT_MUTED = "#8b949e"
    COLOR_ACCENT = "#58a6ff"
    COLOR_ACCENT_HOVER = "#1f6feb"
    COLOR_SUCCESS = "#3fb950"
    COLOR_WARNING = "#d29922"
    COLOR_DANGER = "#f85149"
    COLOR_PURPLE = "#bc8cff"

    def __init__(self):
        super().__init__()
        self.title("AI-Based Sound-to-Visual Translation | ESC-50 Pattern Recognition")
        self.geometry("1280x820")
        self.minsize(1050, 700)
        self.configure(bg=self.COLOR_BG)

        # Paths
        self.project_root = Path(__file__).resolve().parent.parent
        self.datasets_dir = self.project_root / "dataset"
        if not self.datasets_dir.exists():
            self.datasets_dir = Path("K:/AI project/Datasets")

        # State Variables
        self.current_audio_path: Optional[str] = None
        self.current_audio_data: Optional[np.ndarray] = None
        self.current_sr: int = TARGET_SAMPLE_RATE
        self.current_metadata: Optional[Dict[str, Any]] = None
        self.current_prediction: Optional[Dict[str, Any]] = None
        self.current_visual_fig: Optional[plt.Figure] = None
        self.current_visual_params: Optional[Dict[str, Any]] = None

        # Core Engines
        self.player = AudioPlayer()
        self.predictor = AudioPredictor()
        self.translator = SoundVisualTranslator(str(self.project_root / "outputs" / "generated_visuals"))
        self.hardware_info = get_system_hardware_info()

        # Custom Styling
        self._setup_styles()

        # Build UI
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        # Update initial model status
        self.refresh_model_status()

    def _setup_styles(self):
        """Configure TTK modern dark widgets."""
        style = ttk.Style()
        style.theme_use("default")

        # Notebook Tabs
        style.configure("TNotebook", background=self.COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=self.COLOR_CARD,
                        foreground=self.COLOR_TEXT,
                        padding=[16, 8],
                        font=("Segoe UI", 10, "bold"),
                        borderwidth=1,
                        bordercolor=self.COLOR_CARD_BORDER)
        style.map("TNotebook.Tab",
                  background=[("selected", self.COLOR_ACCENT)],
                  foreground=[("selected", "#ffffff")])

        # Progressbar
        style.configure("Horizontal.TProgressbar",
                        background=self.COLOR_ACCENT,
                        troughcolor=self.COLOR_CARD,
                        bordercolor=self.COLOR_CARD_BORDER,
                        lightcolor=self.COLOR_ACCENT,
                        darkcolor=self.COLOR_ACCENT)

    def _build_header(self):
        """Create sleek top navigation and branding banner."""
        header_frame = tk.Frame(self, bg=self.COLOR_CARD, height=60, bd=0, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="AI-Based Sound-to-Visual Translation",
            font=("Segoe UI", 15, "bold"),
            fg=self.COLOR_TEXT_BRIGHT,
            bg=self.COLOR_CARD
        )
        title_lbl.pack(side=tk.LEFT, padx=(20, 10), pady=12)

        subtitle_lbl = tk.Label(
            header_frame,
            text="|  Intelligent Audio Pattern Recognition (ESC-50 CNN)",
            font=("Segoe UI", 10),
            fg=self.COLOR_ACCENT,
            bg=self.COLOR_CARD
        )
        subtitle_lbl.pack(side=tk.LEFT, pady=12)

        # Quick Audio File Indicator
        self.header_file_lbl = tk.Label(
            header_frame,
            text="No Audio Loaded",
            font=("Segoe UI", 9, "italic"),
            fg=self.COLOR_TEXT_MUTED,
            bg=self.COLOR_CARD
        )
        self.header_file_lbl.pack(side=tk.RIGHT, padx=20, pady=12)

    def _build_notebook(self):
        """Build the main 5 tabs container."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 5))

        # Tab 1: Home
        self.tab_home = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(self.tab_home, text="  🏠 Home  ")
        self._init_tab_home()

        # Tab 2: Audio Analysis
        self.tab_analysis = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(self.tab_analysis, text="  📊 Audio Analysis  ")
        self._init_tab_analysis()

        # Tab 3: AI Prediction
        self.tab_prediction = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(self.tab_prediction, text="  🧠 AI Prediction  ")
        self._init_tab_prediction()

        # Tab 4: Visual Translation
        self.tab_visual = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(self.tab_visual, text="  🎨 Visual Translation  ")
        self._init_tab_visual()

        # Tab 5: Model Performance
        self.tab_performance = tk.Frame(self.notebook, bg=self.COLOR_BG)
        self.notebook.add(self.tab_performance, text="  📈 Model Performance  ")
        self._init_tab_performance()

    def _build_statusbar(self):
        """Create bottom status bar."""
        self.status_frame = tk.Frame(self, bg=self.COLOR_CARD, height=26, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_lbl = tk.Label(
            self.status_frame,
            text="Ready. Load an audio file to start signal analysis and AI visual translation.",
            font=("Segoe UI", 9),
            fg=self.COLOR_TEXT_MUTED,
            bg=self.COLOR_CARD
        )
        self.status_lbl.pack(side=tk.LEFT, padx=15, pady=3)

        self.device_lbl = tk.Label(
            self.status_frame,
            text=f"Device: {self.hardware_info['gpu']['device_display']} | PyTorch Backend",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_ACCENT,
            bg=self.COLOR_CARD
        )
        self.device_lbl.pack(side=tk.RIGHT, padx=15, pady=3)

    def set_status(self, text: str, is_error: bool = False, is_success: bool = False):
        """Update the status bar text."""
        fg_col = self.COLOR_DANGER if is_error else (self.COLOR_SUCCESS if is_success else self.COLOR_TEXT_MUTED)
        self.status_lbl.config(text=text, fg=fg_col)
        self.update_idletasks()

    # =========================================================================
    # TAB 1: HOME TAB
    # =========================================================================
    def _init_tab_home(self):
        frame = self.tab_home

        # Left Column: Overview & Quick Actions
        left_col = tk.Frame(frame, bg=self.COLOR_BG)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 10), pady=15)

        # Overview Card
        card_desc = tk.Frame(left_col, bg=self.COLOR_CARD, bd=0, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        card_desc.pack(fill=tk.X, pady=(0, 15), ipady=10)

        tk.Label(card_desc, text="Project Objective & Architecture", font=("Segoe UI", 12, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(5, 8))
        desc_text = (
            "This intelligent desktop application transforms audio recordings into meaningful, deterministic visual geometries.\n\n"
            "• Pipeline: WAV Audio → Preprocessing (Mono, 22.05kHz, Peak Normalization) → Multi-Feature Extraction "
            "(FFT, MFCC, Mel Spectrogram, Spectral Centroid, ZCR, RMS, Chroma) → CNN Pattern Classification (50 ESC-50 Classes) "
            "→ Deterministic Sound-to-Visual Translation Engine → Rendered Art & PNG Export.\n\n"
            "• Key Innovation: Visual shapes are governed by the acoustic physics of the input sound (RMS Energy → Radius, "
            "Spectral Centroid → Rotation, ZCR → Edge Complexity, Dominant Frequency → Harmonic Layering, Class Archetype → Sacred Geometry)."
        )
        tk.Label(card_desc, text=desc_text, font=("Segoe UI", 9), fg=self.COLOR_TEXT, bg=self.COLOR_CARD, justify=tk.LEFT, wraplength=520).pack(anchor="w", padx=15)

        # Workflow Buttons Card
        card_actions = tk.Frame(left_col, bg=self.COLOR_CARD, bd=0, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        card_actions.pack(fill=tk.BOTH, expand=True, ipady=10)

        tk.Label(card_actions, text="Quick Workflow Actions", font=("Segoe UI", 12, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(5, 12))

        btn_grid = tk.Frame(card_actions, bg=self.COLOR_CARD)
        btn_grid.pack(fill=tk.X, padx=15)

        self._create_button(btn_grid, "📂 Load WAV Audio File", self.action_load_audio, col=0, row=0, color=self.COLOR_ACCENT)
        self._create_button(btn_grid, "📊 Open Audio Analysis", lambda: self.notebook.select(1), col=1, row=0)
        self._create_button(btn_grid, "🧠 Run AI Prediction", self.action_run_prediction, col=0, row=1, color=self.COLOR_PURPLE)
        self._create_button(btn_grid, "🎨 Generate Visual Translation", self.action_generate_visual, col=1, row=1, color=self.COLOR_SUCCESS)
        self._create_button(btn_grid, "⚡ Train / Retrain CNN Model", self.action_train_model_dialog, col=0, row=2, color=self.COLOR_WARNING)
        self._create_button(btn_grid, "🔄 Reset Session", self.action_reset, col=1, row=2)

        # Right Column: System Specs & Model Status
        right_col = tk.Frame(frame, bg=self.COLOR_BG, width=420)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 15), pady=15)

        # Hardware Monitor Card
        card_hw = tk.Frame(right_col, bg=self.COLOR_CARD, bd=0, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        card_hw.pack(fill=tk.X, pady=(0, 15), ipady=8)

        tk.Label(card_hw, text="💻 Hardware & Acceleration Environment", font=("Segoe UI", 11, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(5, 8))

        hw = self.hardware_info
        hw_items = [
            ("CPU Processor", f"{hw['cpu']['model']}"),
            ("Logical Cores", f"{hw['cpu']['logical_cores']} Threads"),
            ("System RAM", f"{hw['ram']['formatted']}"),
            ("Primary GPU", f"{hw['gpu']['primary_gpu']}"),
            ("GPU VRAM", f"{hw['gpu']['vram']}"),
            ("Active DL Device", f"{hw['gpu']['device_display']}"),
            ("Keras 3 Engine", f"{hw['keras']['backend'].upper()} Backend (v{hw['keras']['version']}")
        ]
        for label, val in hw_items:
            row_f = tk.Frame(card_hw, bg=self.COLOR_CARD)
            row_f.pack(fill=tk.X, padx=15, pady=2)
            tk.Label(row_f, text=label + ":", font=("Segoe UI", 9, "bold"), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD).pack(side=tk.LEFT)
            tk.Label(row_f, text=val, font=("Segoe UI", 9), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD).pack(side=tk.RIGHT)

        # Model Status Card
        self.card_model = tk.Frame(right_col, bg=self.COLOR_CARD, bd=0, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        self.card_model.pack(fill=tk.BOTH, expand=True, ipady=8)

        tk.Label(self.card_model, text="🤖 Deep Learning Model Status", font=("Segoe UI", 11, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(5, 8))

        self.lbl_model_status_badge = tk.Label(self.card_model, text="Checking status...", font=("Segoe UI", 10, "bold"), fg=self.COLOR_WARNING, bg=self.COLOR_CARD)
        self.lbl_model_status_badge.pack(anchor="w", padx=15, pady=2)

        self.lbl_model_details = tk.Label(self.card_model, text="", font=("Segoe UI", 9), fg=self.COLOR_TEXT, bg=self.COLOR_CARD, justify=tk.LEFT)
        self.lbl_model_details.pack(anchor="w", padx=15, pady=5)

    def refresh_model_status(self):
        """Check if trained model exists and update badge."""
        is_ready = self.predictor.load_resources()
        exp4_meta = self.project_root / "experiments" / "task509_reproduction_experiment4" / "models" / "model_metadata.json"
        default_meta = self.project_root / "models" / "model_metadata.json"
        meta_path = exp4_meta if exp4_meta.exists() else default_meta

        if is_ready:
            self.lbl_model_status_badge.config(text="● MODEL LOADED & READY (ESC-50 CNN - Exp 4: 71.00% Acc)", fg=self.COLOR_SUCCESS)
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    ckpt_name = Path(self.predictor.model_path).name
                    details = (
                        f"• Architecture: 4-Block Conv2D + BatchNorm + GAP\n"
                        f"• Test Accuracy: {meta.get('test_accuracy_pct', 71.0)}% (Fold 5 Held-Out, 284/400 correct)\n"
                        f"• Test Macro Precision: {meta.get('test_precision_pct', 70.62)}% | Recall: {meta.get('test_recall_pct', 71.0)}%\n"
                        f"• Test Macro F1: {meta.get('test_f1_score_pct', 69.24)}%\n"
                        f"• Checkpoint: {ckpt_name} (Epoch 38)\n"
                        f"• Training Device: {meta.get('training_device', 'GPU (NVIDIA RTX 3050)')}"
                    )
                    self.lbl_model_details.config(text=details)
                except Exception:
                    self.lbl_model_details.config(text=f"Model verified: {Path(self.predictor.model_path).name}")
            else:
                self.lbl_model_details.config(text=f"Model file loaded successfully: {Path(self.predictor.model_path).name}")
        else:
            self.lbl_model_status_badge.config(text="○ MODEL NOT TRAINED YET", fg=self.COLOR_WARNING)
            self.lbl_model_details.config(
                text="The CNN classifier has not been trained yet.\n"
                     "Click 'Train / Retrain CNN Model' below or in the\n"
                     "'Model Performance' tab to train the model on ESC-50."
            )

    # =========================================================================
    # TAB 2: AUDIO ANALYSIS TAB
    # =========================================================================
    def _init_tab_analysis(self):
        frame = self.tab_analysis

        # Top Control Bar
        ctrl_bar = tk.Frame(frame, bg=self.COLOR_CARD, bd=0, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        ctrl_bar.pack(fill=tk.X, padx=15, pady=(10, 5), ipady=4)

        tk.Button(ctrl_bar, text="📂 Select WAV File", command=self.action_load_audio, font=("Segoe UI", 9, "bold"),
                  bg=self.COLOR_ACCENT, fg="#ffffff", activebackground=self.COLOR_ACCENT_HOVER, relief=tk.FLAT, padx=12, pady=4).pack(side=tk.LEFT, padx=(10, 10))

        tk.Button(ctrl_bar, text="▶ Play", command=self.action_play_audio, font=("Segoe UI", 9, "bold"),
                  bg=self.COLOR_SUCCESS, fg="#ffffff", relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=3)

        tk.Button(ctrl_bar, text="⏸ Pause", command=self.player.pause, font=("Segoe UI", 9),
                  bg=self.COLOR_CARD, fg=self.COLOR_TEXT, relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=3)

        tk.Button(ctrl_bar, text="⏹ Stop", command=self.player.stop, font=("Segoe UI", 9),
                  bg=self.COLOR_CARD, fg=self.COLOR_DANGER, relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=3)

        self.analysis_file_lbl = tk.Label(ctrl_bar, text="No file selected", font=("Segoe UI", 9, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD)
        self.analysis_file_lbl.pack(side=tk.LEFT, padx=15)

        # Metadata Bar (Horizontal Key Metrics)
        self.meta_bar = tk.Frame(frame, bg=self.COLOR_BG)
        self.meta_bar.pack(fill=tk.X, padx=15, pady=4)
        self.meta_stat_labels = {}

        stat_keys = [
            ("Duration", "5.00 s"),
            ("Sample Rate", "22,050 Hz"),
            ("RMS Energy", "0.0000"),
            ("Peak Amp", "0.0000"),
            ("Centroid", "0.0 Hz"),
            ("ZCR", "0.0000"),
            ("Dominant Freq", "0.0 Hz"),
            ("Dominant Pitch", "N/A")
        ]
        for k, default_val in stat_keys:
            f = tk.Frame(self.meta_bar, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2, ipady=3)
            tk.Label(f, text=k, font=("Segoe UI", 8), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD).pack()
            lbl_v = tk.Label(f, text=default_val, font=("Segoe UI", 9, "bold"), fg=self.COLOR_ACCENT, bg=self.COLOR_CARD)
            lbl_v.pack()
            self.meta_stat_labels[k] = lbl_v

        # 4-Grid Embedded Matplotlib Figure Canvas
        self.fig_analysis, self.axs_analysis = plt.subplots(2, 2, figsize=(11, 6), dpi=100)
        self.fig_analysis.patch.set_facecolor(self.COLOR_BG)
        for ax in self.axs_analysis.flat:
            ax.set_facecolor(self.COLOR_CARD)
            ax.tick_params(colors=self.COLOR_TEXT_MUTED, labelsize=8)

        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, master=frame)
        self.canvas_analysis.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        self._plot_empty_analysis()

    def _plot_empty_analysis(self):
        """Draw placeholder empty grids."""
        titles = ["1. Time-Domain Waveform", "2. FFT Frequency Spectrum", "3. MFCC Feature Matrix", "4. Log-Mel Spectrogram"]
        for ax, t in zip(self.axs_analysis.flat, titles):
            ax.clear()
            ax.set_facecolor(self.COLOR_CARD)
            ax.set_title(t, color=self.COLOR_TEXT_BRIGHT, fontsize=10, weight="bold")
            ax.text(0.5, 0.5, "Load an audio file to visualize", color=self.COLOR_TEXT_MUTED, ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.grid(True, color=self.COLOR_CARD_BORDER, linestyle="--", alpha=0.5)
        self.fig_analysis.tight_layout(pad=2.0)
        self.canvas_analysis.draw()

    def update_audio_analysis_plots(self):
        """Compute and draw the 4-grid real audio features."""
        if self.current_audio_data is None:
            return

        y = self.current_audio_data
        sr = self.current_sr

        # 1. Waveform
        ax1 = self.axs_analysis[0, 0]
        ax1.clear()
        ax1.set_facecolor(self.COLOR_CARD)
        time_axis = np.linspace(0, len(y) / sr, len(y))
        ax1.plot(time_axis, y, color=self.COLOR_ACCENT, linewidth=0.8)
        ax1.set_title("1. Time-Domain Waveform y(t)", color=self.COLOR_TEXT_BRIGHT, fontsize=10, weight="bold")
        ax1.set_xlabel("Time (s)", color=self.COLOR_TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Amplitude", color=self.COLOR_TEXT_MUTED, fontsize=8)
        ax1.set_ylim(-1.05, 1.05)
        ax1.grid(True, color=self.COLOR_CARD_BORDER, linestyle="--", alpha=0.5)

        # 2. FFT Spectrum
        ax2 = self.axs_analysis[0, 1]
        ax2.clear()
        ax2.set_facecolor(self.COLOR_CARD)
        fft_data = compute_fft(y, sr=sr)
        ax2.plot(fft_data["plot_freqs"], fft_data["plot_magnitude_db"], color=self.COLOR_PURPLE, linewidth=1.0)
        ax2.set_title(f"2. FFT Spectrum (Peak: {fft_data['dominant_freq_hz']} Hz)", color=self.COLOR_TEXT_BRIGHT, fontsize=10, weight="bold")
        ax2.set_xlabel("Frequency (Hz)", color=self.COLOR_TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Magnitude (dB)", color=self.COLOR_TEXT_MUTED, fontsize=8)
        ax2.set_xlim(0, 8000)
        ax2.grid(True, color=self.COLOR_CARD_BORDER, linestyle="--", alpha=0.5)

        # 3. MFCC Matrix
        ax3 = self.axs_analysis[1, 0]
        ax3.clear()
        ax3.set_facecolor(self.COLOR_CARD)
        mfcc_data = compute_mfcc(y, sr=sr, n_mfcc=20)
        im3 = ax3.imshow(mfcc_data["mfcc_matrix"], aspect="auto", origin="lower", cmap="magma")
        ax3.set_title("3. MFCC Matrix (20 Coefficients)", color=self.COLOR_TEXT_BRIGHT, fontsize=10, weight="bold")
        ax3.set_xlabel("Time Frames", color=self.COLOR_TEXT_MUTED, fontsize=8)
        ax3.set_ylabel("MFCC Bands", color=self.COLOR_TEXT_MUTED, fontsize=8)

        # 4. Mel Spectrogram
        ax4 = self.axs_analysis[1, 1]
        ax4.clear()
        ax4.set_facecolor(self.COLOR_CARD)
        mel_data = compute_mel_spectrogram(y, sr=sr, n_mels=128, target_frames=216)
        im4 = ax4.imshow(mel_data, aspect="auto", origin="lower", cmap="viridis")
        ax4.set_title("4. Log-Mel Spectrogram (128x216 CNN Input)", color=self.COLOR_TEXT_BRIGHT, fontsize=10, weight="bold")
        ax4.set_xlabel("Time Frames (216)", color=self.COLOR_TEXT_MUTED, fontsize=8)
        ax4.set_ylabel("Mel Bands (128)", color=self.COLOR_TEXT_MUTED, fontsize=8)

        for ax in self.axs_analysis.flat:
            ax.tick_params(colors=self.COLOR_TEXT_MUTED, labelsize=8)

        self.fig_analysis.tight_layout(pad=1.8)
        self.canvas_analysis.draw()

    # =========================================================================
    # TAB 3: AI PREDICTION TAB
    # =========================================================================
    def _init_tab_prediction(self):
        frame = self.tab_prediction

        # Top Action Bar
        top_bar = tk.Frame(frame, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        top_bar.pack(fill=tk.X, padx=15, pady=(10, 10), ipady=5)

        self.btn_run_pred = tk.Button(
            top_bar, text="🧠 Classify Loaded Audio", command=self.action_run_prediction, font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_PURPLE, fg="#ffffff", relief=tk.FLAT, padx=15, pady=6
        )
        self.btn_run_pred.pack(side=tk.LEFT, padx=15)

        self.pred_prog_bar = ttk.Progressbar(top_bar, mode="indeterminate", length=160)

        self.pred_status_lbl = tk.Label(top_bar, text="Ready to predict sound pattern.", font=("Segoe UI", 9), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD)
        self.pred_status_lbl.pack(side=tk.LEFT, padx=10)

        # Main Split Content
        split_frame = tk.Frame(frame, bg=self.COLOR_BG)
        split_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Left: Top-1 Hero Card
        self.hero_card = tk.Frame(split_frame, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        self.hero_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), ipady=15)

        tk.Label(self.hero_card, text="PRIMARY PREDICTION (TOP-1)", font=("Segoe UI", 10, "bold"), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD).pack(anchor="w", padx=20, pady=(10, 5))

        self.lbl_hero_class = tk.Label(self.hero_card, text="—", font=("Segoe UI", 28, "bold"), fg=self.COLOR_ACCENT, bg=self.COLOR_CARD)
        self.lbl_hero_class.pack(anchor="w", padx=20, pady=(5, 2))

        self.lbl_hero_conf = tk.Label(self.hero_card, text="Confidence: — %", font=("Segoe UI", 14, "bold"), fg=self.COLOR_SUCCESS, bg=self.COLOR_CARD)
        self.lbl_hero_conf.pack(anchor="w", padx=20, pady=2)

        self.lbl_hero_badge = tk.Label(self.hero_card, text="", font=("Segoe UI", 9, "bold"), fg=self.COLOR_PURPLE, bg=self.COLOR_CARD)
        self.lbl_hero_badge.pack(anchor="w", padx=20, pady=4)

        self.lbl_hero_latency = tk.Label(self.hero_card, text="Inference Latency: — ms", font=("Segoe UI", 9), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD)
        self.lbl_hero_latency.pack(anchor="w", padx=20, pady=4)

        # Action: Proceed to Visual
        tk.Button(self.hero_card, text="🎨 Proceed to Visual Translation →", command=self.action_generate_visual,
                  font=("Segoe UI", 10, "bold"), bg=self.COLOR_SUCCESS, fg="#ffffff", relief=tk.FLAT, padx=14, pady=8).pack(anchor="w", padx=20, pady=(20, 10))

        # Right: Top-5 Ranking Table
        self.rank_card = tk.Frame(split_frame, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        self.rank_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), ipady=10)

        tk.Label(self.rank_card, text="TOP-5 CLASS RANKINGS & CONFIDENCE", font=("Segoe UI", 11, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD).pack(anchor="w", padx=20, pady=(10, 15))

        self.top5_rows = []
        for i in range(5):
            row_frame = tk.Frame(self.rank_card, bg=self.COLOR_CARD)
            row_frame.pack(fill=tk.X, padx=20, pady=6)

            rank_num = tk.Label(row_frame, text=f"#{i+1}", font=("Segoe UI", 10, "bold"), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD, width=3)
            rank_num.pack(side=tk.LEFT)

            c_name = tk.Label(row_frame, text="—", font=("Segoe UI", 10, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD, width=18, anchor="w")
            c_name.pack(side=tk.LEFT, padx=5)

            pbar = ttk.Progressbar(row_frame, style="Horizontal.TProgressbar", length=140, maximum=100)
            pbar.pack(side=tk.LEFT, padx=5)

            c_pct = tk.Label(row_frame, text="0.0%", font=("Segoe UI", 9, "bold"), fg=self.COLOR_ACCENT, bg=self.COLOR_CARD, width=8, anchor="e")
            c_pct.pack(side=tk.LEFT, padx=5)

            self.top5_rows.append((c_name, pbar, c_pct))

    # =========================================================================
    # TAB 4: VISUAL TRANSLATION TAB
    # =========================================================================
    def _init_tab_visual(self):
        frame = self.tab_visual

        # Left Column: Interactive Visual Canvas
        left_frame = tk.Frame(frame, bg=self.COLOR_BG)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 10), pady=10)

        # Visual Matplotlib Canvas
        self.fig_visual = plt.figure(figsize=(7, 7), dpi=100, facecolor=self.COLOR_BG)
        self.ax_visual = self.fig_visual.add_subplot(111, facecolor=self.COLOR_BG)
        self.ax_visual.set_aspect("equal")
        self.ax_visual.axis("off")
        self.ax_visual.text(0.5, 0.5, "Click 'Generate Visual Translation' to render",
                            color=self.COLOR_TEXT_MUTED, ha="center", va="center", fontsize=11)

        self.canvas_visual = FigureCanvasTkAgg(self.fig_visual, master=left_frame)
        self.canvas_visual.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right Column: Controls & Telemetry Parameters
        right_frame = tk.Frame(frame, bg=self.COLOR_BG, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 15), pady=10)

        # Controls Card
        ctrl_card = tk.Frame(right_frame, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        ctrl_card.pack(fill=tk.X, pady=(0, 12), ipady=8)

        tk.Label(ctrl_card, text="Visual Render Settings", font=("Segoe UI", 11, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(5, 8))

        tk.Label(ctrl_card, text="Generative Mode:", font=("Segoe UI", 9), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD).pack(anchor="w", padx=15)
        self.combo_visual_mode = ttk.Combobox(ctrl_card, values=VISUAL_MODES, state="readonly", font=("Segoe UI", 9))
        self.combo_visual_mode.current(0)
        self.combo_visual_mode.pack(fill=tk.X, padx=15, pady=(2, 10))

        tk.Button(ctrl_card, text="🎨 Render Visual", command=self.action_generate_visual, font=("Segoe UI", 10, "bold"),
                  bg=self.COLOR_SUCCESS, fg="#ffffff", relief=tk.FLAT, padx=12, pady=6).pack(fill=tk.X, padx=15, pady=4)

        tk.Button(ctrl_card, text="💾 Save Visual as PNG", command=self.action_save_visual_png, font=("Segoe UI", 9, "bold"),
                  bg=self.COLOR_ACCENT, fg="#ffffff", relief=tk.FLAT, padx=12, pady=5).pack(fill=tk.X, padx=15, pady=4)

        # Telemetry Card (Physical Feature to Visual Parameter Mapping)
        self.telemetry_card = tk.Frame(right_frame, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        self.telemetry_card.pack(fill=tk.BOTH, expand=True, ipady=8)

        tk.Label(self.telemetry_card, text="Sound-to-Visual Telemetry", font=("Segoe UI", 11, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD).pack(anchor="w", padx=15, pady=(5, 8))

        self.telemetry_labels = {}
        mapping_desc = [
            ("Predicted Class", "Class Archetype Geometry"),
            ("RMS Energy", "Radius & Global Scale"),
            ("Spectral Centroid", "Angular Rotation (θ)"),
            ("Zero Crossing Rate", "Vertices & Edge Lobes"),
            ("Dominant Frequency", "Harmonic Layer Count"),
            ("Dominant Pitch", "Chromatic Color Palette"),
            ("Peak Amplitude", "Waveform Deformation")
        ]
        for feat_name, map_name in mapping_desc:
            row = tk.Frame(self.telemetry_card, bg=self.COLOR_CARD)
            row.pack(fill=tk.X, padx=15, pady=3)
            lbl_title = tk.Label(row, text=f"{feat_name} → {map_name}:", font=("Segoe UI", 8, "bold"), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD, anchor="w")
            lbl_title.pack(side=tk.TOP, anchor="w")
            lbl_val = tk.Label(row, text="—", font=("Segoe UI", 9), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_CARD, anchor="w")
            lbl_val.pack(side=tk.TOP, anchor="w")
            self.telemetry_labels[feat_name] = lbl_val

    # =========================================================================
    # TAB 5: MODEL PERFORMANCE TAB
    # =========================================================================
    def _init_tab_performance(self):
        frame = self.tab_performance

        # Experiment 4 Title & Scope Card
        header_card = tk.Frame(frame, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        header_card.pack(fill=tk.X, padx=15, pady=(10, 4), ipady=5)

        tk.Label(
            header_card,
            text="Experiment 4 Evaluation Results (Held-Out Fold 5 - 400 Samples, 284 Correct)",
            font=("Segoe UI", 11, "bold"),
            fg=self.COLOR_TEXT_BRIGHT,
            bg=self.COLOR_CARD
        ).pack(side=tk.LEFT, padx=15)

        tk.Label(
            header_card,
            text="Checkpoint: audio_cnn_model_task509_best_val_loss.keras (Epoch 38)",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_SUCCESS,
            bg=self.COLOR_CARD
        ).pack(side=tk.RIGHT, padx=15)

        # Top Metrics Cards Row
        top_metrics = tk.Frame(frame, bg=self.COLOR_BG)
        top_metrics.pack(fill=tk.X, padx=15, pady=(4, 8))

        self.perf_metric_labels = {}
        metrics_list = [
            ("Test Accuracy", "71.00%"),
            ("Test Macro Precision", "70.62%"),
            ("Test Macro Recall", "71.00%"),
            ("Test Macro F1-Score", "69.24%"),
            ("Training Device", "GPU (RTX 3050)")
        ]
        for name, def_val in metrics_list:
            f = tk.Frame(top_metrics, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, ipady=6)
            tk.Label(f, text=name, font=("Segoe UI", 8), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_CARD).pack()
            lbl_v = tk.Label(f, text=def_val, font=("Segoe UI", 12, "bold"), fg=self.COLOR_ACCENT, bg=self.COLOR_CARD)
            lbl_v.pack()
            self.perf_metric_labels[name] = lbl_v

        # Action bar
        act_bar = tk.Frame(frame, bg=self.COLOR_CARD, highlightthickness=1, highlightbackground=self.COLOR_CARD_BORDER)
        act_bar.pack(fill=tk.X, padx=15, pady=(0, 10), ipady=4)

        tk.Button(act_bar, text="⚡ Retrain Model on ESC-50 (Folds 1-3 Train / 4 Val / 5 Test)", command=self.action_train_model_dialog,
                  font=("Segoe UI", 9, "bold"), bg=self.COLOR_WARNING, fg="#ffffff", relief=tk.FLAT, padx=12, pady=4).pack(side=tk.LEFT, padx=15)

        tk.Button(act_bar, text="🔄 Reload Performance Graphs", command=self.refresh_performance_tab,
                  font=("Segoe UI", 9), bg=self.COLOR_CARD, fg=self.COLOR_TEXT, relief=tk.FLAT, padx=10, pady=4).pack(side=tk.LEFT, padx=5)

        # Embedded Images Display (Side by Side: Curves + Confusion Matrix)
        self.perf_images_frame = tk.Frame(frame, bg=self.COLOR_BG)
        self.perf_images_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.lbl_img_curves = tk.Label(self.perf_images_frame, text="Training History Curves", bg=self.COLOR_CARD, fg=self.COLOR_TEXT_MUTED)
        self.lbl_img_curves.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.lbl_img_cm = tk.Label(self.perf_images_frame, text="50-Class Confusion Matrix", bg=self.COLOR_CARD, fg=self.COLOR_TEXT_MUTED)
        self.lbl_img_cm.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.refresh_performance_tab()

    def refresh_performance_tab(self):
        """Load and display saved evaluation graphs and metrics from Experiment 4."""
        exp4_dir = self.project_root / "experiments" / "task509_reproduction_experiment4"
        meta_path = exp4_dir / "models" / "model_metadata.json"
        if not meta_path.exists():
            meta_path = self.project_root / "models" / "model_metadata.json"

        graphs_path = exp4_dir / "outputs" / "graphs" / "training_history.png"
        if not graphs_path.exists():
            graphs_path = self.project_root / "outputs" / "graphs" / "training_history.png"

        cm_path = exp4_dir / "outputs" / "confusion_matrix" / "confusion_matrix.png"
        if not cm_path.exists():
            cm_path = self.project_root / "outputs" / "confusion_matrix" / "confusion_matrix.png"

        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                acc = meta.get("test_accuracy_pct", 71.0)
                prec = meta.get("test_precision_pct", 70.62)
                rec = meta.get("test_recall_pct", 71.0)
                f1 = meta.get("test_f1_score_pct", 69.24)
                dev = meta.get("training_device", "GPU (NVIDIA RTX 3050)")

                self.perf_metric_labels["Test Accuracy"].config(text=f"{acc:.2f}%", fg=self.COLOR_SUCCESS)
                self.perf_metric_labels["Test Macro Precision"].config(text=f"{prec:.2f}%")
                self.perf_metric_labels["Test Macro Recall"].config(text=f"{rec:.2f}%")
                self.perf_metric_labels["Test Macro F1-Score"].config(text=f"{f1:.2f}%", fg=self.COLOR_PURPLE)
                self.perf_metric_labels["Training Device"].config(text=str(dev))
            except Exception as e:
                print(f"[refresh_performance_tab] Warning parsing metadata: {e}")

        # Load curves image
        if graphs_path.exists():
            try:
                img_c = Image.open(graphs_path)
                img_c.thumbnail((560, 420))
                self.tk_img_curves = ImageTk.PhotoImage(img_c)
                self.lbl_img_curves.config(image=self.tk_img_curves, text="")
                self.lbl_img_curves.image = self.tk_img_curves
            except Exception:
                self.lbl_img_curves.config(text="Could not load training history image.")
        else:
            self.lbl_img_curves.config(text="No training history available yet.\nTrain the model to view accuracy/loss curves.")

        # Load confusion matrix image
        if cm_path.exists():
            try:
                img_m = Image.open(cm_path)
                img_m.thumbnail((560, 420))
                self.tk_img_cm = ImageTk.PhotoImage(img_m)
                self.lbl_img_cm.config(image=self.tk_img_cm, text="")
                self.lbl_img_cm.image = self.tk_img_cm
            except Exception:
                self.lbl_img_cm.config(text="Could not load confusion matrix image.")
        else:
            self.lbl_img_cm.config(text="No confusion matrix available yet.\nTrain the model to evaluate on Fold 5.")

    # =========================================================================
    # ACTIONS & WORKFLOW LOGIC
    # =========================================================================
    def action_load_audio(self):
        """Open file dialog to pick a WAV audio file."""
        initial_dir = str(self.datasets_dir / "audio") if (self.datasets_dir / "audio").exists() else str(self.project_root)
        file_path = filedialog.askopenfilename(
            title="Select WAV Audio File",
            initialdir=initial_dir,
            filetypes=[("WAV Audio Files", "*.wav"), ("All Audio Files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.set_status(f"Loading audio file: {Path(file_path).name}...")
            y, sr = load_audio(file_path, target_sr=TARGET_SAMPLE_RATE)
            meta = get_audio_metadata(file_path, y, sr)

            self.current_audio_path = file_path
            self.current_audio_data = y
            self.current_sr = sr
            self.current_metadata = meta

            # Update labels
            f_name = Path(file_path).name
            self.header_file_lbl.config(text=f"Loaded: {f_name}")
            self.analysis_file_lbl.config(text=f"{f_name} ({meta['filesize_kb']} KB)")

            # Update stats bar
            self.meta_stat_labels["Duration"].config(text=f"{meta['processed_duration_sec']:.2f} s")
            self.meta_stat_labels["Sample Rate"].config(text=f"{meta['processed_sample_rate']:,} Hz")
            self.meta_stat_labels["RMS Energy"].config(text=f"{meta['rms_energy']:.4f}")
            self.meta_stat_labels["Peak Amp"].config(text=f"{meta['peak_amplitude']:.4f}")

            # Extract quick features for centroid, ZCR, dominant freq & pitch
            v_feats = extract_all_visual_features(y, sr)
            self.meta_stat_labels["Centroid"].config(text=f"{v_feats['raw_centroid_hz']:.1f} Hz")
            self.meta_stat_labels["ZCR"].config(text=f"{v_feats['raw_zcr_mean']:.4f}")
            self.meta_stat_labels["Dominant Freq"].config(text=f"{v_feats['raw_dominant_freq_hz']:.1f} Hz")
            self.meta_stat_labels["Dominant Pitch"].config(text=f"{v_feats['dominant_pitch']}")

            # Load into audio player
            self.player.load(file_path)

            # Update analysis plots
            self.update_audio_analysis_plots()

            self.set_status(f"Successfully loaded '{f_name}'. Ready for pattern recognition.", is_success=True)

        except Exception as e:
            messagebox.showerror("Audio Loading Error", f"Failed to load audio file:\n{e}")
            self.set_status(f"Error loading audio: {e}", is_error=True)

    def action_play_audio(self):
        """Trigger audio playback."""
        if not self.current_audio_path:
            messagebox.showinfo("No Audio", "Please select a WAV audio file first.")
            return
        self.player.play()
        self.set_status("Playing audio...")

    def action_run_prediction(self):
        """Run CNN inference on currently loaded audio in a responsive background thread."""
        if self.current_audio_path is None:
            messagebox.showinfo("No Audio", "Please load a WAV audio file first.")
            return

        if not self.predictor.is_loaded:
            if not self.predictor.load_resources():
                res = messagebox.askyesno(
                    "Model Not Trained",
                    "The CNN model has not been trained yet.\nWould you like to train it now on ESC-50?"
                )
                if res:
                    self.action_train_model_dialog()
                return

        # Disable button, show loading progress indicator
        self.btn_run_pred.config(state="disabled")
        self.pred_prog_bar.pack(side=tk.LEFT, padx=10)
        self.pred_prog_bar.start(10)
        self.pred_status_lbl.config(text="Running CNN inference on audio signal...", fg=self.COLOR_ACCENT)
        f_name = Path(self.current_audio_path).name
        self.set_status(f"Running CNN Audio Pattern Recognition on {f_name}...")

        audio_path = self.current_audio_path

        def worker():
            try:
                pred = self.predictor.predict(audio_path)
                self.after(0, lambda: self._on_prediction_success(pred))
            except Exception as e:
                self.after(0, lambda: self._on_prediction_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_prediction_success(self, pred: Dict[str, Any]):
        """Handle prediction completion on the main UI thread."""
        self.pred_prog_bar.stop()
        self.pred_prog_bar.pack_forget()
        self.btn_run_pred.config(state="normal")
        self.current_prediction = pred

        # Switch to prediction tab
        self.notebook.select(2)

        # Update Hero Card
        p_class = pred["predicted_class"].upper()
        conf = pred["confidence_pct"]
        self.lbl_hero_class.config(text=p_class)
        self.lbl_hero_conf.config(text=f"Confidence: {conf:.1f}%")
        badge_text = "⭐ Official ESC-10 Class" if pred["is_esc10"] else "ESC-50 Class"
        self.lbl_hero_badge.config(text=badge_text)
        self.lbl_hero_latency.config(text=f"Inference Latency: {pred['inference_time_ms']} ms")

        # Update Top-5 Table
        for i, (c_name, c_pct, is_10) in enumerate(pred["top_5"]):
            name_lbl, pbar, pct_lbl = self.top5_rows[i]
            badge = "★ " if is_10 else "   "
            name_lbl.config(text=f"{badge}{c_name}")
            pbar["value"] = c_pct
            pct_lbl.config(text=f"{c_pct:.1f}%")

        self.pred_status_lbl.config(
            text=f"Classified as '{pred['predicted_class']}' with {conf:.1f}% confidence in {pred['inference_time_ms']} ms.",
            fg=self.COLOR_SUCCESS
        )
        self.set_status(f"Prediction complete: {pred['predicted_class']} ({conf:.1f}%)", is_success=True)

    def _on_prediction_error(self, err: Exception):
        """Handle prediction failure on the main UI thread."""
        self.pred_prog_bar.stop()
        self.pred_prog_bar.pack_forget()
        self.btn_run_pred.config(state="normal")
        self.pred_status_lbl.config(text=f"Prediction error: {err}", fg=self.COLOR_DANGER)
        messagebox.showerror("Prediction Error", f"Failed to execute audio classification:\n{err}")
        self.set_status(f"Prediction error: {err}", is_error=True)

    def action_generate_visual(self):
        """Generate and render the Sound-to-Visual artwork."""
        if self.current_audio_data is None:
            messagebox.showinfo("No Audio", "Please select a WAV audio file first.")
            return

        # Ensure prediction exists
        if self.current_prediction is None:
            if self.predictor.is_loaded:
                try:
                    self.current_prediction = self.predictor.predict(self.current_audio_path)
                except Exception:
                    self.current_prediction = {"predicted_class": "sound", "confidence_pct": 100.0}
            else:
                self.current_prediction = {"predicted_class": "sound", "confidence_pct": 100.0}

        try:
            self.set_status("Generating Sound-to-Visual Translation...")
            mode = self.combo_visual_mode.get()
            pred_class = self.current_prediction["predicted_class"]
            conf = self.current_prediction["confidence_pct"]

            # Render visual
            fig, params = self.translator.render_visual(
                y=self.current_audio_data,
                sr=self.current_sr,
                predicted_class=pred_class,
                confidence=conf,
                visual_mode=mode,
                figsize=(6.5, 6.5),
                dpi=100
            )

            self.current_visual_fig = fig
            self.current_visual_params = params

            # Redraw canvas
            self.canvas_visual.figure = fig
            self.canvas_visual.draw()

            # Switch to Visual Translation tab
            self.notebook.select(3)

            # Update Telemetry labels
            self.telemetry_labels["Predicted Class"].config(
                text=f"{params['predicted_class'].upper()} (Archetype: {params['archetype']})",
                fg=self.COLOR_ACCENT
            )
            self.telemetry_labels["RMS Energy"].config(
                text=f"Mean: {params['raw_rms']} → Radius Scale: {params['base_radius']}"
            )
            self.telemetry_labels["Spectral Centroid"].config(
                text=f"{params['raw_centroid']} Hz → Angle: {params['rotation_angle_deg']}°"
            )
            self.telemetry_labels["Zero Crossing Rate"].config(
                text=f"Mean: {params['raw_zcr']} → Vertices / Lobes: {params['vertices']}"
            )
            self.telemetry_labels["Dominant Frequency"].config(
                text=f"{params['raw_dom_freq']} Hz → Layers: {params['harmonic_layers']}"
            )
            self.telemetry_labels["Dominant Pitch"].config(
                text=f"Note: {params['pitch_note']} → Primary Hue: {params['hue_primary']}"
            )
            self.telemetry_labels["Peak Amplitude"].config(
                text=f"Deformation Scale: {params['deformation_scale']}"
            )

            self.set_status(f"Visual artwork generated successfully using '{mode}'.", is_success=True)

        except Exception as e:
            messagebox.showerror("Visual Translation Error", f"Failed to generate visual translation:\n{e}")
            self.set_status(f"Visual translation error: {e}", is_error=True)

    def action_save_visual_png(self):
        """Save generated visual as high-resolution PNG."""
        if self.current_visual_fig is None:
            messagebox.showinfo("No Visual", "Please generate a visual translation first.")
            return

        pred_class = self.current_visual_params.get("predicted_class", "sound") if self.current_visual_params else "sound"
        save_path = filedialog.asksaveasfilename(
            title="Save Visual Translation as PNG",
            initialdir=str(self.translator.output_dir),
            initialfile=f"visual_{pred_class}_{time.strftime('%Y%m%d_%H%M%S')}.png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            saved_file = self.translator.save_visual_png(self.current_visual_fig, predicted_class=pred_class, custom_filename=save_path)
            messagebox.showinfo("Visual Saved", f"Successfully exported high-resolution PNG image to:\n{saved_file}")
            self.set_status(f"Saved visual to: {saved_file}", is_success=True)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save visual image:\n{e}")

    def action_train_model_dialog(self):
        """Launch background training thread with a modal progress window."""
        res = messagebox.askyesno(
            "Start CNN Training",
            "This will train the 2D CNN on ESC-50 dataset splits:\n"
            "• Folds 1, 2, 3: Training (1,200 audio files)\n"
            "• Fold 4: Validation (400 audio files)\n"
            "• Fold 5: Final Testing (400 audio files)\n\n"
            "Feature caching will ensure lightning-fast preprocessing.\n"
            "Proceed with training now?"
        )
        if not res:
            return

        # Modal progress popup
        prog_win = tk.Toplevel(self)
        prog_win.title("Training ESC-50 CNN Model")
        prog_win.geometry("520x220")
        prog_win.configure(bg=self.COLOR_BG)
        prog_win.transient(self)
        prog_win.grab_set()

        tk.Label(prog_win, text="⚡ Training CNN Audio Classifier", font=("Segoe UI", 12, "bold"), fg=self.COLOR_TEXT_BRIGHT, bg=self.COLOR_BG).pack(pady=(15, 5))

        lbl_stage = tk.Label(prog_win, text="Initializing training engine...", font=("Segoe UI", 9), fg=self.COLOR_ACCENT, bg=self.COLOR_BG)
        lbl_stage.pack(pady=5)

        pbar = ttk.Progressbar(prog_win, style="Horizontal.TProgressbar", length=450, maximum=100)
        pbar.pack(pady=10)

        lbl_details = tk.Label(prog_win, text="Detecting hardware & cached features...", font=("Segoe UI", 8), fg=self.COLOR_TEXT_MUTED, bg=self.COLOR_BG)
        lbl_details.pack(pady=5)

        def progress_cb(status_text: str, frac: float):
            self.after(0, lambda: [
                lbl_stage.config(text=status_text),
                pbar.config(value=frac * 100),
                lbl_details.config(text=f"Progress: {frac*100:.1f}%")
            ])

        def worker():
            try:
                train_audio_classifier(
                    dataset_dir=str(self.datasets_dir),
                    epochs=30,
                    initial_batch_size=self.hardware_info["recommended_batch_size"],
                    progress_callback=progress_cb
                )
                self.after(0, lambda: [
                    prog_win.destroy(),
                    self.predictor.load_resources(),
                    self.refresh_model_status(),
                    self.refresh_performance_tab(),
                    self.notebook.select(4),
                    messagebox.showinfo("Training Completed", "CNN model trained and evaluated successfully on Fold 5!")
                ])
            except Exception as err:
                self.after(0, lambda: [
                    prog_win.destroy(),
                    messagebox.showerror("Training Error", f"Training encountered an error:\n{err}")
                ])

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def action_reset(self):
        """Clear loaded audio and reset visualizations."""
        self.player.stop()
        self.current_audio_path = None
        self.current_audio_data = None
        self.current_metadata = None
        self.current_prediction = None
        self.current_visual_fig = None
        self.current_visual_params = None

        self.header_file_lbl.config(text="No Audio Loaded")
        self.analysis_file_lbl.config(text="No file selected")
        self.lbl_hero_class.config(text="—")
        self.lbl_hero_conf.config(text="Confidence: — %")
        self.lbl_hero_badge.config(text="")
        self.lbl_hero_latency.config(text="Inference Latency: — ms")

        for name_lbl, pbar, pct_lbl in self.top5_rows:
            name_lbl.config(text="—")
            pbar["value"] = 0
            pct_lbl.config(text="0.0%")

        for k in self.meta_stat_labels:
            self.meta_stat_labels[k].config(text="—")

        for k in self.telemetry_labels:
            self.telemetry_labels[k].config(text="—")

        self._plot_empty_analysis()
        self.ax_visual.clear()
        self.ax_visual.set_facecolor(self.COLOR_BG)
        self.ax_visual.text(0.5, 0.5, "Click 'Generate Visual Translation' to render",
                            color=self.COLOR_TEXT_MUTED, ha="center", va="center", fontsize=11)
        self.canvas_visual.draw()
        self.set_status("Session reset.", is_success=True)

    def _create_button(self, parent, text, cmd, col=0, row=0, color=None):
        """Helper to create sleek styled buttons."""
        bg_col = color if color else self.COLOR_CARD_BORDER
        btn = tk.Button(
            parent,
            text=text,
            command=cmd,
            font=("Segoe UI", 9, "bold"),
            bg=bg_col,
            fg="#ffffff" if color else self.COLOR_TEXT_BRIGHT,
            activebackground=self.COLOR_ACCENT_HOVER,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=8,
            cursor="hand2"
        )
        btn.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        return btn


def launch_gui():
    """Main application entrypoint."""
    app = SoundVisualizerApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
