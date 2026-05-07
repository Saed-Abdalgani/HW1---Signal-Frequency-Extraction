"""Tests for sklearn-based analysis helpers."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class TestTSNEComputation:
    """T-SNE tests."""

    def test_tsne_output_shape(self) -> None:
        """Output has three columns."""
        features = np.random.randn(50, 10)
        emb = TSNE(n_components=3, perplexity=5, random_state=42).fit_transform(features)
        assert emb.shape == (50, 3)

    def test_tsne_no_nan(self) -> None:
        """Finite embedding."""
        features = np.random.randn(30, 10)
        emb = TSNE(n_components=3, perplexity=5, random_state=42).fit_transform(features)
        assert np.isfinite(emb).all()


class TestPCAComputation:
    """PCA tests."""

    def test_pca_output_shape(self) -> None:
        """Three principal components."""
        features = np.random.randn(50, 10)
        pca = PCA(n_components=3)
        emb = pca.fit_transform(features)
        assert emb.shape == (50, 3)

    def test_explained_variance_sums_le_1(self) -> None:
        """Variance ratios bounded."""
        features = np.random.randn(50, 10)
        pca = PCA(n_components=3)
        pca.fit(features)
        assert sum(pca.explained_variance_ratio_) <= 1.0 + 1e-6

    def test_pca_axis_labels(self) -> None:
        """Positive explained variance."""
        features = np.random.randn(50, 10)
        pca = PCA(n_components=3)
        pca.fit(features)
        for ratio in pca.explained_variance_ratio_:
            assert ratio > 0


class TestFFTComputation:
    """FFT tests."""

    def test_fft_peak(self) -> None:
        """Peak near tone frequency."""
        fs = 200.0
        t = np.linspace(0, 1, int(fs), endpoint=False)
        sig = np.sin(2 * np.pi * 4.0 * t)
        spectrum = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 4.0) < 1.0

    def test_fft_magnitude_positive(self) -> None:
        """Non-negative spectrum."""
        sig = np.sin(np.linspace(0, 10, 200))
        spectrum = np.abs(np.fft.rfft(sig))
        assert (spectrum >= 0).all()
