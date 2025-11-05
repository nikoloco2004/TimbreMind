"""Audio analysis utilities for the TimbreMind project.

This module focuses on extracting a few intuitive descriptors from a
``.wav`` file that later drive the visual flow field.  The processing is
intentionally lightweight and heavily documented so that the math behind
the metrics stays approachable to newcomers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile


@dataclass
class AudioFeatures:
    """Container for the audio descriptors used by the visualization.

    Attributes
    ----------
    sample_rate:
        Number of audio samples that represent one second of sound.
    signal:
        Normalised mono signal in the range ``[-1, 1]``.
    rms_envelope:
        Root-mean-square (RMS) energy per analysis window.  Provides a
        smooth approximation of loudness.
    spectral_centroid:
        Brightness descriptor that captures where the "centre of mass"
        of the spectrum lies for each window.  Higher values imply
        brighter, more high-frequency rich content.
    time_axis:
        Time stamps (in seconds) corresponding to each descriptor window.
    """

    sample_rate: int
    signal: np.ndarray
    rms_envelope: np.ndarray
    spectral_centroid: np.ndarray
    time_axis: np.ndarray


def load_wav(path: Path) -> Tuple[int, np.ndarray]:
    """Load a ``.wav`` file and return the sample rate with a mono signal.

    A ``wav`` file may contain stereo data or more exotic bit depths.  The
    ``scipy.io.wavfile`` helper handles these intricacies and returns a
    NumPy array.  The rest of the function ensures the result is
    normalised to floating point ``[-1, 1]`` and, if necessary, mixes
    multi-channel audio down to mono.
    """

    sample_rate, data = wavfile.read(path)

    # Convert integers to floating point while preserving dynamics.  Many
    # ``wav`` files store values as signed 16-bit integers.
    if data.dtype.kind in {"i", "u"}:  # integer or unsigned integer
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    # For stereo files we average the channels.  The arithmetic mean keeps
    # the energy balanced while avoiding clipping.
    if data.ndim > 1:
        data = data.mean(axis=1)

    return sample_rate, data


def sliding_window(signal: np.ndarray, window_size: int, hop: int) -> np.ndarray:
    """Create an efficient view of ``signal`` split into overlapping windows.

    ``numpy.lib.stride_tricks.sliding_window_view`` is used underneath to
    avoid copying data, which keeps the operation fast even for long
    signals.  The resulting shape is ``(num_windows, window_size)``.
    """

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(signal, window_shape=window_size)[::hop]
    return windows.copy()  # copy so downstream operations stay explicit


def compute_rms(windows: np.ndarray) -> np.ndarray:
    """Compute the root-mean-square energy for each analysis window."""

    return np.sqrt(np.mean(np.square(windows), axis=1))


def compute_spectral_centroid(
    windows: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Estimate the spectral centroid for each analysis window.

    A discrete Fourier transform (DFT) is applied to the windowed audio.
    The spectral centroid can then be computed as a weighted average of
    frequency bins where the weights are the magnitudes of the spectrum.
    """

    window_size = windows.shape[1]
    # ``np.fft.rfft`` computes the positive frequency bins of the DFT for
    # real-valued signals, which is exactly what we need.
    spectrum = np.fft.rfft(windows * np.hanning(window_size), axis=1)
    magnitudes = np.abs(spectrum)

    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    numerator = np.sum(magnitudes * freqs, axis=1)
    denominator = np.sum(magnitudes, axis=1) + 1e-12  # avoid division by zero
    return numerator / denominator


def extract_audio_features(path: Path, window_duration: float = 0.05) -> AudioFeatures:
    """Load ``path`` and derive descriptors that drive the flow field.

    Parameters
    ----------
    path:
        Location of the ``.wav`` file to analyse.
    window_duration:
        Length in seconds of the analysis window.  The default of 50 ms
        offers a good compromise between temporal resolution and the
        stability required for smooth visuals.
    """

    sample_rate, signal = load_wav(path)
    window_size = max(1, int(window_duration * sample_rate))
    hop = max(1, window_size // 2)

    # Pad the end of the signal so that the sliding windows cover the
    # entire file.  ``mode="reflect"`` prevents abrupt discontinuities.
    padded = np.pad(signal, (0, window_size), mode="reflect")
    windows = sliding_window(padded, window_size=window_size, hop=hop)

    rms = compute_rms(windows)
    centroid = compute_spectral_centroid(windows, sample_rate)

    # Generate the timeline so each descriptor can be mapped back to the
    # original audio in seconds.
    time_axis = np.arange(len(rms)) * (hop / sample_rate)

    return AudioFeatures(
        sample_rate=sample_rate,
        signal=signal,
        rms_envelope=rms,
        spectral_centroid=centroid,
        time_axis=time_axis,
    )


__all__ = [
    "AudioFeatures",
    "extract_audio_features",
]
