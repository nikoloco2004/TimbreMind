"""Tiny Tkinter screen that plays audio while animating a Perlin flow field.

The goal of this module is to stay as beginner-friendly as possible.  Every
step of the interface is documented in plain English so that somebody new to
GUI programming can follow along and tweak the behaviour.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygame
import tkinter as tk
from tkinter import filedialog, messagebox

from .audio import AudioFeatures, extract_audio_features
from .perlin import PerlinNoise


@dataclass
class CanvasLine:
    """Small helper structure keeping track of a flow line on the canvas."""

    item_id: int  # Tkinter integer handle returned by ``create_line``
    anchor: Tuple[float, float]  # fixed starting position of the line


class FlowFieldViewer:
    """Minimal viewer widget that reacts to audio in real time."""

    def __init__(self, root: tk.Tk) -> None:
        # Store the window handle so we can schedule animation callbacks with ``after``.
        self.root = root
        self.root.title("TimbreMind Live Viewer")
        self.root.configure(bg="#202020")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # These attributes are populated once the user chooses a file.
        self.features: Optional[AudioFeatures] = None  # audio descriptors
        self.audio_buffer: Optional[np.ndarray] = None  # 16-bit PCM data for playback
        self.analysis_times: Optional[np.ndarray] = None  # timeline of descriptor frames
        self.normalised_rms: Optional[np.ndarray] = None  # loudness in the range [0, 1]
        self.normalised_centroid: Optional[np.ndarray] = None  # brightness in [0, 1]

        # ``pygame.mixer`` gives back ``Sound`` and ``Channel`` objects for playback.
        self.sound: Optional[pygame.mixer.Sound] = None
        self.channel: Optional[pygame.mixer.Channel] = None
        # ``is_playing`` mirrors the actual state so the UI can disable updates quickly.
        self.is_playing: bool = False
        # ``play_start`` stores the timestamp at which playback began, letting us estimate
        # how far along the song we are without complex callbacks.
        self.play_start: float = 0.0

        # Set up the control buttons and canvas where the animation is shown.
        self._build_controls()
        self.canvas = self._build_canvas()

        # Prepare a grid of anchors where the lines will be drawn.
        self.canvas_width = 640
        self.canvas_height = 360
        self.grid_columns = 18
        self.grid_rows = 12
        self.lines: List[CanvasLine] = []
        self._seed_lines()

        # The noise generator supplies the directions used by the flow field.
        self.noise = PerlinNoise(seed=0)
        # ``time_offset`` is incremented each frame to gently move the field even when
        # the audio is calm.
        self.time_offset = 0.0

    # ------------------------------------------------------------------ layout --
    def _build_controls(self) -> None:
        """Create a simple bar with file selection and transport buttons."""

        bar = tk.Frame(self.root, bg="#202020")
        bar.pack(side=tk.TOP, fill=tk.X, padx=12, pady=12)

        load_button = tk.Button(
            bar,
            text="Load WAV",
            command=self.choose_file,
            bg="#3a3a3a",
            fg="white",
        )
        load_button.pack(side=tk.LEFT)

        start_button = tk.Button(
            bar,
            text="Start",
            command=self.start_playback,
            bg="#2e7d32",
            fg="white",
        )
        start_button.pack(side=tk.LEFT, padx=8)

        stop_button = tk.Button(
            bar,
            text="Stop",
            command=self.stop_playback,
            bg="#c62828",
            fg="white",
        )
        stop_button.pack(side=tk.LEFT)

        # Informational labels keep the user aware of what is happening.
        self.file_label = tk.StringVar(value="No file loaded yet.")
        file_info = tk.Label(bar, textvariable=self.file_label, bg="#202020", fg="white")
        file_info.pack(side=tk.LEFT, padx=12)

        self.status_label = tk.StringVar(value="Pick a WAV file to get started.")
        status = tk.Label(self.root, textvariable=self.status_label, bg="#202020", fg="#dddddd")
        status.pack(side=tk.TOP, fill=tk.X)

    def _build_canvas(self) -> tk.Canvas:
        """Create the drawing surface that hosts the animated flow field."""

        canvas = tk.Canvas(
            self.root,
            width=640,
            height=360,
            bg="black",
            highlightthickness=0,
        )
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=12)
        return canvas

    def _seed_lines(self) -> None:
        """Generate grid-placed line segments that will be animated later."""

        self.lines.clear()
        self.canvas.delete("all")

        # Even spacing keeps the field tidy and easy to read.
        x_spacing = self.canvas_width / (self.grid_columns + 1)
        y_spacing = self.canvas_height / (self.grid_rows + 1)

        for row in range(self.grid_rows):
            for col in range(self.grid_columns):
                anchor_x = (col + 1) * x_spacing
                anchor_y = (row + 1) * y_spacing
                # Draw an initial short line.  The coordinates will be overwritten during
                # the animation updates.
                item_id = self.canvas.create_line(
                    anchor_x,
                    anchor_y,
                    anchor_x,
                    anchor_y,
                    fill="#5555ff",
                    width=2,
                    capstyle=tk.ROUND,
                )
                self.lines.append(CanvasLine(item_id=item_id, anchor=(anchor_x, anchor_y)))

    # -------------------------------------------------------------- user input --
    def choose_file(self) -> None:
        """Ask the user for a ``.wav`` file and compute the descriptors."""

        file_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            features = extract_audio_features(Path(file_path))
        except Exception as exc:  # pragma: no cover - friendly dialog for manual use
            messagebox.showerror("Audio error", f"Could not load file:\n{exc}")
            return

        # Stop anything that might currently be playing so we do not overlap sounds.
        self.stop_playback()

        self.features = features
        self.file_label.set(Path(file_path).name)
        self.status_label.set("File ready. Press Start to play.")

        # Normalise the descriptors so that mapping into motion/colour is easy.
        self.normalised_rms = self._normalise(features.rms_envelope)
        self.normalised_centroid = self._normalise(features.spectral_centroid)

        # Extend the descriptor timeline with a final value to avoid interpolation gaps
        # at the very end of the song.
        duration = len(features.signal) / features.sample_rate
        times = features.time_axis
        if times.size == 0 or times[-1] < duration:
            times = np.append(times, duration)
            if self.normalised_rms.size:
                self.normalised_rms = np.append(self.normalised_rms, self.normalised_rms[-1])
            if self.normalised_centroid.size:
                self.normalised_centroid = np.append(
                    self.normalised_centroid, self.normalised_centroid[-1]
                )
        self.analysis_times = times

        # Convert the floating point mono signal into 16-bit PCM expected by ``pygame``.
        clipped = np.clip(features.signal, -1.0, 1.0)
        self.audio_buffer = (clipped * 32767).astype(np.int16)

    def start_playback(self) -> None:
        """Play the loaded audio and kick off the animation loop."""

        if self.features is None or self.audio_buffer is None:
            messagebox.showinfo("Missing audio", "Please load a WAV file first.")
            return
        if self.is_playing:
            return  # already running

        # Ensure there are no hanging sounds from a previous run.
        self.stop_playback()

        # ``pygame.mixer`` needs to be initialised with the correct audio format.
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        try:
            pygame.mixer.init(
                frequency=int(self.features.sample_rate),
                size=-16,  # signed 16-bit samples
                channels=1,
            )
        except pygame.error as exc:  # pragma: no cover - surface runtime error
            messagebox.showerror("Audio error", f"Could not start playback:\n{exc}")
            return

        # Create a ``Sound`` object from the raw PCM bytes and start playing it.
        self.sound = pygame.mixer.Sound(buffer=self.audio_buffer.tobytes())
        self.channel = self.sound.play()
        if self.channel is None:  # pragma: no cover - defensive guard
            messagebox.showerror("Audio error", "Could not start playback.")
            pygame.mixer.quit()
            self.sound = None
            return

        self.is_playing = True
        self.play_start = time.perf_counter()
        self.status_label.set("Playing...")
        self.time_offset = 0.0

        # Start the animation immediately.
        self.root.after(0, self.update_visual)

    def stop_playback(self) -> None:
        """Stop the current playback and freeze the animation."""

        if self.channel is not None:
            self.channel.stop()
            self.channel = None
        self.sound = None
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        if self.is_playing:
            self.status_label.set("Playback stopped.")
        self.is_playing = False

    # --------------------------------------------------------------- animation --
    def update_visual(self) -> None:
        """Refresh the positions and colours of every line on the canvas."""

        if (
            not self.is_playing
            or self.features is None
            or self.channel is None
            or self.analysis_times is None
            or self.normalised_rms is None
            or self.normalised_centroid is None
        ):
            return

        # ``pygame`` exposes ``get_busy`` so we know when the sound is done.
        if not self.channel.get_busy():
            self.finish_playback()
            return

        elapsed = time.perf_counter() - self.play_start
        lookup_time = min(elapsed, float(self.analysis_times[-1]))

        # Use ``np.interp`` to sample the descriptor envelopes at the current time.
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

        # Step the noise field.  Louder sounds push the flow faster.
        self.time_offset += 0.03 + loudness * 0.15
        base_scale = 0.012 + brightness * 0.02
        line_length = 12 + loudness * 28

        for entry in self.lines:
            start_x, start_y = entry.anchor
            noise_value = float(
                self.noise(
                    np.array([start_x * base_scale]),
                    np.array([(start_y + self.time_offset) * base_scale]),
                )[0]
            )
            # ``PerlinNoise`` returns values around [-1, 1].  Map that range onto full angles.
            angle = (noise_value + 1.0) * math.pi
            end_x = start_x + math.cos(angle) * line_length
            end_y = start_y + math.sin(angle) * line_length
            self.canvas.coords(entry.item_id, start_x, start_y, end_x, end_y)
            self.canvas.itemconfigure(entry.item_id, fill=self._colour_from_brightness(brightness))

        # Schedule the next frame roughly 30 times per second.
        self.root.after(33, self.update_visual)

    def finish_playback(self) -> None:
        """Handle natural completion when the audio buffer runs out."""

        self.is_playing = False
        self.channel = None
        self.sound = None
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        self.status_label.set("Playback finished.")

    # ---------------------------------------------------------------- utilities --
    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        """Scale ``values`` to the ``[0, 1]`` range.  Returns zeros if constant."""

        if values.size == 0:
            return values
        minimum = float(values.min())
        maximum = float(values.max())
        if math.isclose(minimum, maximum):
            return np.zeros_like(values)
        return (values - minimum) / (maximum - minimum)

    @staticmethod
    def _colour_from_brightness(level: float) -> str:
        """Map the ``brightness`` descriptor into a simple blue→orange gradient."""

        level = float(np.clip(level, 0.0, 1.0))
        # Linearly blend between dark blue and warm orange.
        start = np.array([40, 90, 255])
        end = np.array([255, 180, 60])
        rgb = (start + (end - start) * level).astype(int)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def on_close(self) -> None:
        """Make sure playback stops cleanly when the window is closed."""

        self.stop_playback()
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        self.root.destroy()


def run() -> None:
    """Entry point used by ``python -m timbremind.ui``."""

    root = tk.Tk()
    FlowFieldViewer(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual interactive usage
    run()
