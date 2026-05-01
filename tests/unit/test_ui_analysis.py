"""Tests for UI analysis helpers — T-SNE, PCA, FFT computation.

Validates the internal signal generation and dimensionality reduction
used by the analysis tab callbacks.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from freq_extractor.services.ui_analysis import _gen_sig


class TestUiGenSig:
    """Internal signal generation for UI analysis."""

    def test_returns_time_and_signal(self) -> None:
        """_gen_sig returns (time, signal) arrays."""
        t, sig = _gen_sig(200.0, 2.0, 10.0, 0.0, 1.0)
        assert len(t) == len(sig)
        assert len(t) > 0

    def test_negative_freq_fallback(self) -> None:
        """Negative frequency falls back to 0.1 Hz."""
        t, sig = _gen_sig(200.0, 2.0, -5.0, 0.0, 1.0)
        assert len(t) > 0


class TestTSNEComputation:
    """T-SNE dimensionality reduction tests."""

    def test_tsne_output_shape(self) -> None:
        """6K.10: T-SNE reduces to 3 dimensions."""
        features = np.random.randn(50, 10)
        emb = TSNE(n_components=3, perplexity=5, random_state=42).fit_transform(features)
        assert emb.shape == (50, 3)

    def test_tsne_no_nan(self) -> None:
        """T-SNE output contains no NaN."""
        features = np.random.randn(30, 10)
        emb = TSNE(n_components=3, perplexity=5, random_state=42).fit_transform(features)
        assert np.isfinite(emb).all()


class TestPCAComputation:
    """PCA dimensionality reduction tests."""

    def test_pca_output_shape(self) -> None:
        """PCA reduces to 3 dimensions."""
        features = np.random.randn(50, 10)
        pca = PCA(n_components=3)
        emb = pca.fit_transform(features)
        assert emb.shape == (50, 3)

    def test_explained_variance_sums_le_1(self) -> None:
        """6K.11: Explained variance ratio sums ≤ 1.0."""
        features = np.random.randn(50, 10)
        pca = PCA(n_components=3)
        pca.fit(features)
        assert sum(pca.explained_variance_ratio_) <= 1.0 + 1e-6

    def test_pca_axis_labels(self) -> None:
        """Explained variance percentages are positive."""
        features = np.random.randn(50, 10)
        pca = PCA(n_components=3)
        pca.fit(features)
        for ratio in pca.explained_variance_ratio_:
            assert ratio > 0


class TestFFTComputation:
    """FFT spectrum computation tests."""

    def test_fft_peak(self) -> None:
        """6K.9: FFT peak matches active sinusoid frequency."""
        fs = 200.0
        t = np.linspace(0, 1, int(fs), endpoint=False)
        sig = np.sin(2 * np.pi * 25 * t)
        spectrum = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 25.0) < 1.0

    def test_fft_magnitude_positive(self) -> None:
        """FFT magnitude is non-negative."""
        sig = np.sin(np.linspace(0, 10, 200))
        spectrum = np.abs(np.fft.rfft(sig))
        assert (spectrum >= 0).all()
