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
