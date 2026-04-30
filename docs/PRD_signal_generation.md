# PRD: Signal Generation Subsystem

| Field    | Value               |
|----------|---------------------|
| Version  | 1.00                |
| Parent   | docs/PRD.md         |
| Updated  | 2026-04-28          |

---

## 1. Description

The Signal Generation subsystem is responsible for producing synthetic sinusoidal signals with configurable frequency, amplitude, phase, noise level, and duration.  It also constructs the sliding-window dataset used for training and evaluation of all three neural network architectures.

---

## 2. Theoretical Background

**Fourier Decomposition**  
Any continuous periodic signal can be expressed as a sum of sinusoids (Fourier series).  Here we isolate the problem to pure single-frequency sinusoids so ground-truth labels are exact.

**Signal Formula**
```
s(t) = A · sin(2π · f · t + φ)
s_noisy(t) = s(t) + ε,   ε ~ N(0, (σ · A)²)
```

**Nyquist–Shannon Sampling Theorem**  
To uniquely reconstruct a signal with maximum frequency f_max, the sampling rate f_s must satisfy:
```
f_s ≥ 2 · f_max    (Nyquist criterion)
```
We choose f_s = 200 Hz for f_max = 50 Hz → safety factor of 4×.

**Sliding Window**  
A context window of W=10 samples is slid over the full signal with step=1, yielding `N - W` dataset entries per signal (where N = total samples = f_s × duration).

---

## 3. Frequency Selection Rationale

| Label | Frequency | Period   | Samples/Period | Reason                                      |
|-------|-----------|----------|----------------|---------------------------------------------|
| f₁    | 5 Hz      | 200 ms   | 40             | Very low-frequency; long-period challenge    |
| f₂    | 15 Hz     | 66.7 ms  | 13.3           | Low-medium; tests moderate memory            |
| f₃    | 30 Hz     | 33.3 ms  | 6.7            | Medium-high; near Nyquist midpoint           |
| f₄    | 50 Hz     | 20 ms    | 4              | High-frequency; hardest within our bandwidth |

Rationale for this spread:
- Spans 1.0–0.5 octave increments for good diversity.
- No frequency is an exact integer multiple of another (avoids harmonic aliasing confusion).
- The range 5–50 Hz covers signals analogous to EEG, vibration, and audio sub-bands.

---

## 4. Inputs

| Parameter       | Type    | Range / Default   | Source               |
|-----------------|---------|-------------------|----------------------|
| frequencies     | list[float] | [5,15,30,50] Hz | config/setup.json    |
| amplitude       | float   | > 0, default 1.0  | config/setup.json    |
| sampling_rate   | int     | ≥ 2×f_max, default 200 | config/setup.json |
| duration_s      | float   | > 0, default 10.0 | config/setup.json    |
| noise_std_ratio | float   | [0, ∞), default 0.10 | config/setup.json |
| window_size     | int     | ≥ 1, default 10   | config/setup.json    |
| seed            | int     | any int, default 42 | CLI / config       |

---

## 5. Outputs

| Output              | Type         | Description                                      |
|---------------------|--------------|--------------------------------------------------|
| raw_signals         | dict[str, ndarray] | {freq_label: clean signal of shape (N,)}   |
| noisy_signals       | dict[str, ndarray] | {freq_label: noisy signal of shape (N,)}   |
| dataset             | list[dict]   | List of {frequency_label, noisy_samples, clean_samples, target_output} |
| split_datasets      | tuple        | (train, val, test) lists                         |
| saved .npz files    | file         | data/signals.npz, data/dataset.npz               |

---

## 6. Expected Behavior

- Each frequency produces exactly `f_s × duration_s` samples (2000 at default settings).
- Sliding window of size 10 → `2000 - 10 = 1990` entries per frequency.
- Total dataset size: 4 × 1990 = 7960 entries.
- Train/val/test: 5572 / 1194 / 1194 entries (approx 70/15/15).
- Phase φ is drawn from Uniform(0, 2π) using the fixed seed.
- Signal values are normalized to zero-mean, unit-variance before training.

---

## 7. Normalization Strategy

Signals are normalized per-frequency using the training set statistics (mean, std) computed from clean samples.  The same normalization is applied to validation and test sets to prevent data leakage.

---

## 8. Performance Metrics for This Subsystem

| Metric                        | Requirement               |
|-------------------------------|---------------------------|
| Signal generation time        | < 1 second                |
| Dataset construction time     | < 5 seconds               |
| Memory footprint              | < 100 MB                  |
| SNR of noisy signal (σ=0.1)   | ≈ 20 dB                   |

---

## 9. Alternatives Considered

| Alternative          | Rejected Because                                             |
|----------------------|--------------------------------------------------------------|
| Real audio data      | Non-reproducible, requires external data source              |
| Multi-frequency composite | Harder to define ground truth; spec requires isolated  |
| Random-walk signals  | Not aligned with Fourier theory objective                    |
| FFT-based generation | Numpy's sin is simpler and avoids spectral leakage artifacts |

---

## 10. Constraints

- Phase must be random (not fixed) to prevent model from memorizing a single sinusoid trajectory.
- Noise is additive Gaussian, not multiplicative (spec requirement).
- Dataset must be persisted to `data/` so training doesn't re-generate every run.

---

## 11. Test Scenarios

| Test ID | Scenario                              | Success Criterion                          |
|---------|---------------------------------------|--------------------------------------------|
| SG-T1   | Generate signal at f=5 Hz, fs=200     | FFT peak at 5 Hz ± 0.5 Hz                 |
| SG-T2   | Noisy signal σ=0.1                    | SNR ≈ 20 dB                               |
| SG-T3   | Window size 10 on 2000-sample signal  | 1990 dataset entries produced              |
| SG-T4   | Seed reproducibility                  | Two runs with seed=42 produce identical data |
| SG-T5   | Edge case: σ=0 (no noise)             | Noisy = clean signal exactly              |
| SG-T6   | Edge case: window_size = total_samples | Should raise ValueError                  |
| SG-T7   | Split ratios sum to 1.0               | len(train)+len(val)+len(test) == total    |

---

## 12. Success Criteria

1. FFT of generated signal shows dominant peak at exactly the configured frequency.
2. Noise is zero-mean and its standard deviation matches `σ × A`.
3. Dataset entries are correctly formatted with one-hot encoding matching the frequency.
4. Reproducible with fixed seed across platforms.
