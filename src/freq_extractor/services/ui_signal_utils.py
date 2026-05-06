"""Signal generation and processing utilities for UI callbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


def gen_signal(fs: float, n_cyc: float, freq: float, phase: float, amp: float) -> tuple:
    """Return time array and sinusoid using legacy duration ``n_cyc / freq``."""
    if freq <= 0:
        freq = 0.1
    duration = n_cyc / freq
    n_samp = max(int(fs * duration), 2)
    t = np.linspace(0, duration, n_samp, endpoint=False)
    return t, amp * np.sin(2 * np.pi * freq * t + phase)


def clamp_n_cycles(n: float | int | None) -> int:
    """Clamp N-cycles to ``[1, 10]``."""
    v = int(n) if n is not None else 3
    return max(1, min(10, v))


def compute_shared_grid(fs: float, n_cyc: float | int | None, f_min: float) -> np.ndarray:
    """Symmetric time axis centred at ``t = 0`` spanning one full-duration window."""
    fm = max(float(f_min), 0.1)
    nc = clamp_n_cycles(n_cyc)
    duration = nc / fm
    n_samp = max(int(float(fs) * duration), 2)
    dt = duration / float(n_samp)
    return (np.arange(n_samp, dtype=float) + 0.5) * dt - duration / 2.0


def f_min_from_sin_bundle(sin_vals: tuple[Any, ...]) -> float:
    """Smallest positive frequency among MIX-enabled sinusoids."""
    freqs: list[float] = []
    for i in range(4):
        b = i * 6
        mix, freq = sin_vals[b], sin_vals[b + 2]
        if mix and "mix" in mix and freq and float(freq) > 0:
            freqs.append(float(freq))
    return min(freqs) if freqs else 1.0


def weighted_mean_freq(sin_vals: tuple[Any, ...]) -> float:
    """Amplitude-weighted mean frequency for combined-signal filtering."""
    num, den = 0.0, 0.0
    for i in range(4):
        b = i * 6
        mix, freq, amp = sin_vals[b], sin_vals[b + 2], sin_vals[b + 4]
        if not (mix and "mix" in mix and freq and float(freq) > 0):
            continue
        a = float(amp or 1.0)
        num += float(freq) * a
        den += a
    return num / den if den > 0 else 1.0


def signal_on_grid(t: np.ndarray, freq: float, phase: float, amp: float) -> np.ndarray:
    """Sinusoidal tone on arbitrary time grid *t*.

    Uses :math:`A\\sin(2\\pi f t + \\phi + \\pi/2)` (equivalently
    :math:`A\\cos(2\\pi f t + \\phi)`), so each wavelength is mirror-symmetric
    through its crest/trough axes
    and, with *t* centred at zero (:func:`compute_shared_grid`), the trace is even
    about ``t = 0`` when :math:`\\phi \\equiv 0 \\pmod{\\pi}`.
    """
    f = max(float(freq), 0.1)
    ph = float(phase or 0.0)
    return float(amp or 1.0) * np.sin(2 * np.pi * f * t + ph + np.pi / 2.0)


def apply_bpf(sig: np.ndarray, freq: float, bw: float, fs: float) -> np.ndarray:
    """Narrow bandpass (zero-phase); used when per-sin BPF is checked."""
    low, high = max(freq - bw / 2, 0.1), min(freq + bw / 2, fs / 2 - 0.1)
    if low >= high or high <= 0:
        return sig
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def apply_filter(
    sig: np.ndarray,
    filt: str | None,
    fs: float,
    center_hz: float,
    bw_hz: float,
) -> np.ndarray:
    """Apply Filter dropdown (zero-phase ``sosfiltfilt`` for display)."""
    if not filt or filt == "None" or len(sig) < 4:
        return sig
    c = max(float(center_hz), 0.1)
    bw = max(float(bw_hz or 5.0), 0.5)
    nyq = fs / 2 - 0.2
    try:
        if filt == "Bandpass":
            lo, hi = max(c - bw / 2, 0.1), min(c + bw / 2, nyq)
            if lo >= hi:
                return sig
            sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
        elif filt == "Lowpass":
            cut = min(c + bw / 2, nyq)
            if cut <= 0.2:
                return sig
            sos = butter(4, cut, btype="low", fs=fs, output="sos")
        elif filt == "Highpass":
            cut = max(c - bw / 2, 0.2)
            if cut >= nyq:
                return sig
            sos = butter(4, cut, btype="high", fs=fs, output="sos")
        else:
            return sig
        return sosfiltfilt(sos, sig)
    except ValueError:
        return sig


def add_noise(sig: np.ndarray, noise_type: str, rng: np.random.Generator) -> np.ndarray:
    """Add noise from GLOBAL PARAMETERS noise dropdown."""
    out = np.asarray(sig, dtype=float)
    if noise_type == "Gaussian":
        out = out + rng.normal(0, 0.2, size=out.shape)
    elif noise_type == "Uniform":
        out = out + rng.uniform(-0.2, 0.2, size=out.shape)
    return out
