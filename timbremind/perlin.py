"""Minimal implementation of smooth Perlin noise used for the flow field."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PerlinNoise:
    """Generate repeatable 2D Perlin noise values.

    The class exposes ``__call__`` so instances behave like simple
    functions.  Seeding the permutation table makes the noise field
    deterministic, which is useful for reproducibility.
    """

    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # ``np.arange(256)`` is shuffled to build the permutation table
        # popularised by Ken Perlin.
        self.permutation = np.arange(256, dtype=int)
        rng.shuffle(self.permutation)
        self.permutation = np.tile(self.permutation, 2)

    @staticmethod
    def fade(t: np.ndarray) -> np.ndarray:
        """Smooth fade curve used by Perlin noise.

        The polynomial ``6t^5 - 15t^4 + 10t^3`` ensures that the first and
        second derivatives vanish at the lattice points, avoiding harsh
        transitions between neighbouring gradients.
        """

        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Linearly interpolate between ``a`` and ``b`` using factor ``t``."""

        return a + t * (b - a)

    @staticmethod
    def gradient(hash_value: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Map ``hash_value`` to a pseudo-random gradient vector.

        Only four gradient directions are required for 2D noise.  The
        ``hash_value`` indexes into the set via bitwise operations.
        """

        h = hash_value & 3
        u = np.where(h < 2, x, y)
        v = np.where(h < 2, y, x)
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return noise values for the coordinate arrays ``x`` and ``y``."""

        # Convert coordinates to the unit grid and gather corner indices.
        xi = np.floor(x).astype(int) & 255
        yi = np.floor(y).astype(int) & 255

        xf = x - np.floor(x)
        yf = y - np.floor(y)

        u = self.fade(xf)
        v = self.fade(yf)

        aaa = self.permutation[self.permutation[xi] + yi]
        aba = self.permutation[self.permutation[xi] + yi + 1]
        baa = self.permutation[self.permutation[xi + 1] + yi]
        bba = self.permutation[self.permutation[xi + 1] + yi + 1]

        x1 = self.lerp(
            self.gradient(aaa, xf, yf),
            self.gradient(baa, xf - 1, yf),
            u,
        )
        x2 = self.lerp(
            self.gradient(aba, xf, yf - 1),
            self.gradient(bba, xf - 1, yf - 1),
            u,
        )

        return self.lerp(x1, x2, v)


def generate_flow_angles(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    noise: PerlinNoise,
    scale: float,
    time_factor: float,
) -> np.ndarray:
    """Convert Perlin noise samples into angles for the flow field.

    ``scale`` controls how quickly the noise changes over space.  Larger
    values reveal smaller, more detailed patterns.  ``time_factor`` is an
    offset applied to the ``y`` coordinate that allows the animation to be
    modulated by the audio descriptors.
    """

    nx = grid_x * scale
    ny = (grid_y + time_factor) * scale
    noise_values = noise(nx, ny)
    return np.mod(noise_values * 2 * math.pi, 2 * math.pi)


__all__ = ["PerlinNoise", "generate_flow_angles"]
