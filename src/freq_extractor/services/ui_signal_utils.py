"""Signal generation and processing utilities for UI callbacks.

Extracted from ui_callbacks.py for code organization.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def gen_signal(fs: float, n_cyc: float, freq: float, phase: float, amp: float) -> tuple:
    """Generate a time array and sinusoidal signal."""
    if freq <= 0:
        freq = 0.1
    duration = n_cyc / freq
    n_samp = max(int(fs * duration), 2)
    t = np.linspace(0, duration, n_samp, endpoint=False)
    return t, amp * np.sin(2 * np.pi * freq * t + phase)


def apply_bpf(sig: np.ndarray, freq: float, bw: float, fs: float) -> np.ndarray:
    """Apply a bandpass filter centred at *freq* with width *bw*."""
    low, high = max(freq - bw / 2, 0.1), min(freq + bw / 2, fs / 2 - 0.1)
    if low >= high or high <= 0:
        return sig
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, sig)


def apply_filter(sig: np.ndarray, freq: float, bw: float, fs: float, mode: str) -> np.ndarray:
    """Apply the selected per-component filter around *freq*."""
    if mode == "None" or not mode:
        return sig
    nyq = fs / 2
    low, high = max(freq - bw / 2, 0.1), min(freq + bw / 2, nyq - 0.1)
    if mode == "Bandpass":
        return apply_bpf(sig, freq, bw, fs)
    if mode == "Lowpass" and high < nyq:
        return sosfilt(butter(4, high, btype="lowpass", fs=fs, output="sos"), sig)
    if mode == "Highpass" and 0 < low < nyq:
        return sosfilt(butter(4, low, btype="highpass", fs=fs, output="sos"), sig)
    return sig


def add_noise(sig: np.ndarray, noise_type: str, rng: np.random.Generator) -> np.ndarray:
    """Add global noise to signal based on type selection."""
    std = 0.2
    if noise_type == "Gaussian":
        return sig + rng.normal(0, std, size=sig.shape)
    if noise_type == "Uniform":
        return sig + rng.uniform(-std, std, size=sig.shape)
    return sig

from typing import Any

def compute_shared_grid(fs: float, n_cyc: float, min_freq: float) -> np.ndarray:
    """Generate a symmetric time grid centered at 0."""
    if min_freq <= 0:
        min_freq = 0.1
    duration = n_cyc / min_freq
    n_samp = max(int(fs * duration), 2)
    dt = duration / n_samp
    return (np.arange(n_samp) - (n_samp - 1) / 2) * dt

def signal_on_grid(t: np.ndarray, freq: float, phase: float, amp: float) -> np.ndarray:
    """Generate a sine wave with a pi/2 phase shift for symmetry on the grid."""
    return amp * np.sin(2 * np.pi * freq * t + phase + np.pi / 2)

def clamp_n_cycles(n_cyc: Any) -> float:
    """Clamp the number of cycles to a float."""
    try:
        return float(n_cyc)
    except (TypeError, ValueError):
        return 5.0

def f_min_from_sin_bundle(sin_vals: tuple) -> float:
    """Find the minimum frequency of active sinusoids in the bundle."""
    freqs = []
    for i in range(4):
        mix = sin_vals[i * 6]
        freq = sin_vals[i * 6 + 2]
        if mix and "mix" in mix and freq and float(freq) > 0:
            freqs.append(float(freq))
    return min(freqs) if freqs else 0.1

def weighted_mean_freq(freqs: list[float], amps: list[float]) -> float:
    """Calculate the weighted mean of the given frequencies."""
    if not freqs or sum(amps) == 0:
        return 0.1
    return sum(f * a for f, a in zip(freqs, amps)) / sum(amps)

