"""Tkinter user interface that plays audio while animating the flow field."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import simpleaudio as sa
import tkinter as tk
from tkinter import filedialog, messagebox

from .audio import AudioFeatures, extract_audio_features
from .perlin import PerlinNoise, generate_flow_angles


class RealtimeVisualizerApp:
    """Minimal Tkinter application showing the live flow field animation.

    The widget layout keeps the requirements intentionally simple: a button to
    select a ``.wav`` file, start/stop controls, and a canvas that visualises
    how the Perlin-noise field reacts to the playing audio.  All attributes are
    thoroughly documented so newcomers can understand the data that flows
    through the UI.
    """

    def __init__(self, master: tk.Tk) -> None:
        # Store the Tk root window so callbacks can schedule work via ``after``.
        self.master = master
        self.master.title("TimbreMind Live Flow Field")
        self.master.configure(bg="#1e1e1e")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # ``AudioFeatures`` computed for the currently loaded file.
        self.features: Optional[AudioFeatures] = None
        # Normalised descriptors cached to avoid recomputing within the update
        # loop.  The arrays are extended to match the full audio duration.
        self.normalised_rms: Optional[np.ndarray] = None
        self.normalised_centroid: Optional[np.ndarray] = None
        self.analysis_times: Optional[np.ndarray] = None

        # ``simpleaudio`` playback handle.  ``None`` when no audio is playing.
        self.playback: Optional[sa.PlayObject] = None
        # Buffer that holds the PCM representation of the currently loaded
        # signal.  Defined upfront so attribute access stays predictable even
        # before the first file is selected.
        self.audio_buffer: Optional[np.ndarray] = None
        # Timestamp captured right after starting playback so elapsed time can
        # be estimated without querying audio callbacks.
        self.play_start_time: float = 0.0
        # Flag used to short-circuit the update loop when the user stops
        # playback manually or closes the window.
        self.is_playing: bool = False

        # Dimensions of the animated canvas.  Smaller values keep rendering
        # light-weight while still showcasing the flow behaviour.
        self.canvas_width = 640
        self.canvas_height = 400
        # Grid resolution controlling how many arrows appear in the flow field.
        self.grid_columns = 24
        self.grid_rows = 15

        # Build the layout: a compact control bar followed by the matplotlib
        # canvas that hosts the quiver plot.
        self._build_controls()
        self._build_canvas()

        # ``PerlinNoise`` instance reused for every frame so the permutation
        # table stays consistent throughout the animation.
        self.noise = PerlinNoise(seed=0)
        # Pre-compute the grid positions (X and Y coordinates) once.  These
        # remain constant so the update loop only needs to change the vector
        # directions.
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(0, self.canvas_width, self.grid_columns),
            np.linspace(0, self.canvas_height, self.grid_rows),
        )
        # Small random offsets add colour variation between neighbouring
        # arrows.  ``default_rng`` produces reproducible values.
        rng = np.random.default_rng(42)
        self.colour_offsets = rng.random(self.grid_x.shape)
        self.colormap = cm.magma

        # Initialise the quiver plot with zero-length vectors so the canvas
        # displays a subtle idle state before audio is loaded.
        zeros = np.zeros_like(self.grid_x)
        self.quiver = self.ax.quiver(
            self.grid_x,
            self.grid_y,
            zeros,
            zeros,
            color=self.colormap(np.zeros_like(self.grid_x)).reshape(-1, 4),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            linewidth=1.5,
        )
        self.canvas.draw()

    # ------------------------------------------------------------------ UI ----
    def _build_controls(self) -> None:
        """Create the file selection and transport controls."""

        control_frame = tk.Frame(self.master, bg="#1e1e1e")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=12)

        load_button = tk.Button(
            control_frame,
            text="Load WAV",
            command=self.choose_file,
            bg="#3a3a3a",
            fg="white",
            relief=tk.RAISED,
        )
        load_button.pack(side=tk.LEFT)

        start_button = tk.Button(
            control_frame,
            text="Start",
            command=self.start_playback,
            bg="#2f7d32",
            fg="white",
            relief=tk.RAISED,
        )
        start_button.pack(side=tk.LEFT, padx=8)

        stop_button = tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_playback,
            bg="#b71c1c",
            fg="white",
            relief=tk.RAISED,
        )
        stop_button.pack(side=tk.LEFT)

        # Labels keep the user informed about the loaded file and the current
        # state of the animation.
        self.file_label = tk.StringVar(value="No file selected.")
        file_display = tk.Label(
            control_frame,
            textvariable=self.file_label,
            bg="#1e1e1e",
            fg="white",
        )
        file_display.pack(side=tk.LEFT, padx=12)

        self.status_label = tk.StringVar(value="Load a WAV file to begin.")
        status_display = tk.Label(
            self.master,
            textvariable=self.status_label,
            bg="#1e1e1e",
            fg="#d0d0d0",
        )
        status_display.pack(side=tk.TOP, fill=tk.X)

    def _build_canvas(self) -> None:
        """Initialise the matplotlib figure embedded inside Tk."""

        canvas_frame = tk.Frame(self.master, bg="#1e1e1e")
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.figure = Figure(figsize=(6.4, 4.0), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("black")
        self.ax.set_xlim(0, self.canvas_width)
        self.ax.set_ylim(0, self.canvas_height)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Perlin Flow Field", color="white")

        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------- actions ----
    def choose_file(self) -> None:
        """Prompt the user for a ``.wav`` file and extract audio features."""

        file_path = filedialog.askopenfilename(
            title="Select a WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            features = extract_audio_features(Path(file_path))
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror(
                "Audio error",
                f"Could not load {file_path}.\n\n{exc}",
            )
            return

        # Stop any previous playback before loading new data.
        self.stop_playback()

        self.features = features
        self.file_label.set(Path(file_path).name)
        self.status_label.set("File loaded. Press Start to play.")

        # Normalise descriptors so they can directly modulate colours and
        # motion.  ``math.isclose`` guards against division by zero when the
        # feature is constant.
        self.normalised_rms = self._normalise(features.rms_envelope)
        self.normalised_centroid = self._normalise(features.spectral_centroid)
        # ``AudioFeatures.time_axis`` covers the descriptor frames.  Extending
        # it ensures that interpolation remains valid until the exact end of
        # the audio signal.
        duration = len(features.signal) / features.sample_rate
        times = features.time_axis
        if times[-1] < duration:
            times = np.append(times, duration)
            self.normalised_rms = np.append(self.normalised_rms, self.normalised_rms[-1])
            self.normalised_centroid = np.append(
                self.normalised_centroid, self.normalised_centroid[-1]
            )
        self.analysis_times = times

        # Convert the floating point audio to 16-bit PCM which
        # ``simpleaudio`` expects.  ``np.clip`` avoids integer overflow.
        self.audio_buffer = np.clip(features.signal, -1.0, 1.0)
        self.audio_buffer = (self.audio_buffer * 32767).astype(np.int16)

    def start_playback(self) -> None:
        """Begin audio playback and kick off the visual update loop."""

        if self.features is None or self.audio_buffer is None:
            messagebox.showinfo("No audio", "Please load a WAV file first.")
            return
        if self.is_playing:
            return

        # Ensure previous audio is fully stopped before starting again.
        self.stop_playback()

        # ``simpleaudio`` plays the buffer asynchronously, returning a handle
        # that lets us query the playback state.
        self.playback = sa.play_buffer(
            self.audio_buffer,
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=self.features.sample_rate,
        )
        self.play_start_time = time.perf_counter()
        self.is_playing = True
        self.status_label.set("Playing...")

        # Schedule the first frame immediately.  Subsequent frames re-schedule
        # themselves while playback is active.
        self.master.after(0, self.update_visual)

    def stop_playback(self) -> None:
        """Stop audio and freeze the animation."""

        if self.playback is not None:
            self.playback.stop()
            self.playback = None
        if self.is_playing:
            self.status_label.set("Playback stopped.")
        self.is_playing = False

    # ----------------------------------------------------------- animation ----
    def update_visual(self) -> None:
        """Refresh the quiver plot so it reflects the current audio frame."""

        if (
            not self.is_playing
            or self.features is None
            or self.playback is None
            or self.analysis_times is None
            or self.normalised_rms is None
            or self.normalised_centroid is None
        ):
            return

        # ``simpleaudio`` exposes ``is_playing`` so we can stop gracefully when
        # the buffer finishes without requiring an explicit callback.
        if not self.playback.is_playing():
            self.finish_playback()
            return

        elapsed = time.perf_counter() - self.play_start_time
        # Clamp the lookup time to the descriptor range to keep ``np.interp``
        # well-defined even after the very last frame.
        lookup_time = min(elapsed, self.analysis_times[-1])
        loudness = float(
            np.interp(
                lookup_time,
                self.analysis_times,
                self.normalised_rms,
                left=self.normalised_rms[0],
                right=self.normalised_rms[-1],
            )
        )
        brightness = float(
            np.interp(
                lookup_time,
                self.analysis_times,
                self.normalised_centroid,
                left=self.normalised_centroid[0],
                right=self.normalised_centroid[-1],
            )
        )

        # ``time_factor`` shifts the noise lookup over time.  Louder moments
        # sweep the offset faster to convey energy, while the base speed keeps
        # the field moving even during quiet passages.
        time_factor = elapsed * 1.5 + loudness * 8.0
        scale = 0.015 + brightness * 0.035
        angles = generate_flow_angles(
            self.grid_x,
            self.grid_y,
            noise=self.noise,
            scale=scale,
            time_factor=time_factor,
        )

        magnitude = 0.6 + loudness * 1.8
        u = np.cos(angles) * magnitude
        v = np.sin(angles) * magnitude

        colour_values = np.clip(brightness * 0.7 + self.colour_offsets * 0.3, 0.0, 1.0)
        colours = self.colormap(colour_values)

        self.quiver.set_UVC(u, v)
        self.quiver.set_color(colours.reshape(-1, 4))
        self.canvas.draw_idle()

        # Request another frame roughly 30 times per second while playback is
        # active.
        self.master.after(33, self.update_visual)

    def finish_playback(self) -> None:
        """Handle natural playback completion triggered by the audio buffer."""

        self.is_playing = False
        self.playback = None
        self.status_label.set("Playback finished.")

    # --------------------------------------------------------------- utils ----
    @staticmethod
    def _normalise(feature: np.ndarray) -> np.ndarray:
        """Return ``feature`` scaled to the ``[0, 1]`` range."""

        if feature.size == 0:
            return feature
        minimum = float(feature.min())
        maximum = float(feature.max())
        if math.isclose(minimum, maximum):
            return np.zeros_like(feature)
        return (feature - minimum) / (maximum - minimum)

    def on_close(self) -> None:
        """Stop playback and destroy the window when the user exits."""

        self.stop_playback()
        self.master.destroy()


def run() -> None:
    """Launch the Tkinter visualiser."""

    root = tk.Tk()
    RealtimeVisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - interactive entry point
    run()
