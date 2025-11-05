"""Flow field visualisation orchestrating the Perlin noise and audio data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .audio import AudioFeatures
from .perlin import PerlinNoise, generate_flow_angles


@dataclass
class FlowFieldConfig:
    """Parameters controlling the appearance of the generated artwork."""

    seed: int = 0
    resolution: Tuple[int, int] = (1920, 1080)
    grid_size: Tuple[int, int] = (80, 45)
    step_size: float = 4.0
    steps_per_line: int = 200
    line_width: float = 1.0


class FlowFieldRenderer:
    """Create flow field images guided by audio descriptors."""

    def __init__(self, config: FlowFieldConfig) -> None:
        self.config = config
        self.noise = PerlinNoise(seed=config.seed)

    def _normalise_feature(self, feature: np.ndarray) -> np.ndarray:
        """Scale a feature to the ``[0, 1]`` range for visual modulation."""

        if feature.size == 0:
            return feature
        min_val = feature.min()
        max_val = feature.max()
        if math.isclose(min_val, max_val):
            return np.zeros_like(feature)
        return (feature - min_val) / (max_val - min_val)

    def _seed_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the initial positions of each line in the flow field."""

        width, height = self.config.resolution
        grid_x = np.linspace(0, width, self.config.grid_size[0])
        grid_y = np.linspace(0, height, self.config.grid_size[1])
        return np.meshgrid(grid_x, grid_y)

    def _build_colours(
        self, loudness: np.ndarray, brightness: np.ndarray
    ) -> np.ndarray:
        """Construct RGBA colours using loudness for opacity and hue shifts."""

        # Hue is derived from the spectral centroid (brightness), wrapped so
        # that brighter timbres rotate around the colour wheel.
        hues = (brightness * 0.75 + 0.1) % 1.0
        # Loudness (RMS) controls the opacity.  Softer moments fade lines out.
        alphas = np.clip(loudness * 0.85 + 0.15, 0.0, 1.0)

        # Convert HSV to RGB using ``matplotlib`` utilities.
        colours = np.array([plt.cm.hsv(h) for h in hues])
        colours[:, 3] = alphas
        return colours

    def render(
        self,
        features: AudioFeatures,
        output_path: Path,
    ) -> None:
        """Draw the flow field as a PNG image and write it to ``output_path``."""

        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        width, height = self.config.resolution
        xs, ys = self._seed_points()

        loudness = self._normalise_feature(features.rms_envelope)
        brightness = self._normalise_feature(features.spectral_centroid)

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_facecolor("black")
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis("off")

        num_lines = xs.size
        flattened_xs = xs.ravel()
        flattened_ys = ys.ravel()

        # Stretch the descriptors so that one value roughly corresponds to a
        # single flow line.  ``np.interp`` handles the mapping elegantly.
        descriptor_indices = np.linspace(0, len(loudness) - 1, num_lines)
        loudness_for_lines = np.interp(descriptor_indices, np.arange(len(loudness)), loudness)
        brightness_for_lines = np.interp(
            descriptor_indices, np.arange(len(brightness)), brightness
        )
        colours_for_lines = self._build_colours(loudness_for_lines, brightness_for_lines)

        for idx, (x0, y0, colour, loud) in enumerate(
            zip(flattened_xs, flattened_ys, colours_for_lines, loudness_for_lines)
        ):
            # ``time_factor`` offsets the Perlin noise lookup.  Louder
            # segments explore more of the noise volume, creating richer
            # motion in energetic sections.
            time_factor = loud * 5.0
            # Start the polyline from the seed point and walk following the
            # angle field.  The number of steps is modulated by the loudness
            # to make quiet moments shorter and calmer.
            steps = int(self.config.steps_per_line * (0.3 + loud * 0.7))
            positions = np.zeros((steps, 2), dtype=float)
            positions[0] = [x0, y0]

            for step in range(1, steps):
                angle = generate_flow_angles(
                    grid_x=np.array([positions[step - 1, 0]]),
                    grid_y=np.array([positions[step - 1, 1]]),
                    noise=self.noise,
                    scale=0.01,
                    time_factor=time_factor + step * 0.01,
                )[0]
                dx = math.cos(angle) * self.config.step_size
                dy = math.sin(angle) * self.config.step_size
                positions[step] = positions[step - 1] + (dx, dy)

                # Keep the points inside the canvas.  When a path escapes the
                # image bounds we stop drawing further segments.
                if not (0 <= positions[step, 0] <= width and 0 <= positions[step, 1] <= height):
                    positions = positions[:step]
                    break

            ax.plot(
                positions[:, 0],
                positions[:, 1],
                color=colour,
                linewidth=self.config.line_width * (0.5 + loud),
                alpha=colour[3],
            )

        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(output_path, dpi=100, facecolor="black", transparent=False)
        plt.close(fig)


__all__ = ["FlowFieldConfig", "FlowFieldRenderer"]
