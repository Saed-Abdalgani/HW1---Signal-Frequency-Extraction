# Product Requirements Document (PRD)
## HW1 — Signal Frequency Extraction using Neural Networks

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Version       | 1.20                                       |
| Status        | Approved                                   |
| Author        | Saed Abdalgani                             |
| Date          | 2026-04-28                                 |
| Package       | `freq_extractor`                           |

---

## 1. Project Purpose

Implement, train, and compare three neural network architectures — a Fully Connected Multi-Layer Perceptron (MLP), a Recurrent Neural Network (RNN), and a Long Short-Term Memory network (LSTM) — for the task of extracting a target frequency component from a composite noisy signal.  The project demonstrates how network memory depth and gating mechanisms affect ability to denoise time-series data.

---

## 2. User Problem

Given a noisy, multi-frequency time-series signal it is difficult to isolate a single frequency component without explicit frequency-domain transforms (e.g., FFT).  Neural networks that learn temporal patterns should be able to learn this decomposition implicitly.  Understanding which architecture best captures short vs. long temporal dependencies in signals is a core competency in applied deep learning.

---

## 3. Target Users

| User           | Context                                                    |
|----------------|------------------------------------------------------------|
| Course grader  | Evaluates code quality, correctness, and documentation     |
| ML practitioner| Reuses code as a template for time-series problems         |
| Researcher     | Reproduces and extends experiments                         |

---

## 4. Goals

1. Generate a reproducible, well-defined synthetic dataset of composite noisy signals.
2. Train all three architectures (MLP, RNN, LSTM) to predict the next clean sample of the target frequency.
3. Achieve ≤ 0.05 MSE on the test set for at least one architecture.
4. Quantitatively compare architectures across multiple noise levels and frequencies.
5. Produce a professional, submission-ready lab report and reproducible codebase.
6. Deliver an interactive **Sinusoid Explorer** browser dashboard allowing real-time signal composition, noise/filter control, and four visualization tabs (Signals, T-SNE 3D, PCA 3D, FFT Spectrum).

---

## 5. Key Performance Indicators (KPIs)

| KPI                          | Target                          |
|------------------------------|---------------------------------|
| Test MSE (best model)        | < 0.05                          |
| Test coverage                | ≥ 85 %                          |
| Ruff violations              | 0                               |
| Max source file length       | ≤ 145 code lines (v1.10)        |
| Training convergence         | All 3 models converge           |
| Reproducibility              | Same seed → identical results   |
| UI launch                    | `--mode ui` opens on localhost   |

---

## 6. Acceptance Criteria

- [x] All three networks train without runtime errors.
- [x] At least one model reaches test MSE < 0.05.
- [x] All models exhibit distinct characteristics matching theoretical expectations.
- [x] Training curves, prediction plots, and comparison tables are saved.
- [x] `uv run pytest tests/ --cov=src` passes with ≥ 85 % coverage.
- [x] `uv run ruff check .` reports 0 violations.
- [x] README serves as a self-contained lab report.
- [x] Results are reproducible with `SEED=42`.
- [x] `uv run python src/main.py --mode ui` opens the Sinusoid Explorer on http://localhost:8050.

---

## 7. Functional Requirements

### FR-1: Signal Generation
- Generate pure sinusoidal signals: `s(t) = A·sin(2π·f·t + φ)`.
- Add Gaussian noise: `s_noisy(t) = s(t) + ε`, where `ε ~ N(0, (σ·A)²)`.
- Support four configurable frequencies: f₁=5 Hz, f₂=15 Hz, f₃=30 Hz, f₄=50 Hz.
- Sampling rate: 200 Hz (≥ 2 × 50 Hz, configured in `config/setup.json`).
- Duration: 10 seconds per frequency → 2000 samples per signal.
- Random phase shift φ ∈ [0, 2π) per signal instance.
- Each sinusoid has independently configurable amplitude A ∈ [0, 2.0] and phase φ ∈ [0, 2π).

### FR-2: Dataset Construction
- Sliding window of size 10 over each signal.
- Each entry: `{frequency_label (one-hot, len=4), noisy_samples (10,), clean_samples (10,), target_output (1,)}`.
- Split: 70 % train / 15 % validation / 15 % test (stratified by frequency).
- Persist dataset to `data/` as `.npz` files.
- All random operations use a configurable global seed (default 42).

### FR-3: MLP Model
- Input: concatenated [10 noisy samples ‖ 4-dim one-hot] = 14-dimensional vector.
- Architecture: 14 → 64 → 128 → 64 → 1 with Tanh activations.
- Output: scalar prediction of next clean sample.

### FR-4: RNN Model
- Input per timestep: [noisy_sample_t ‖ one-hot] = 5-dimensional vector.
- Sequence length: 10 timesteps.
- Hidden size: 64 units, 2 layers.
- Output: final hidden state → Linear → scalar.

### FR-5: LSTM Model
- Same input structure as RNN.
- LSTM cells with input/forget/output gates.
- Hidden size: 64, 2 layers.
- Output: final hidden state → Linear → scalar.

### FR-6: Training
- Loss: Mean Squared Error (MSE).
- Optimizer: Adam (lr=0.001, configurable).
- Batch size: 64 (configurable).
- Max epochs: 300 with early stopping (patience=20).
- Save best checkpoint per model.
- Log train/validation MSE per epoch.

### FR-7: Evaluation
- Compute train/validation/test MSE for each model.
- Plot training curves (loss vs. epoch).
- Plot example predictions vs. clean ground truth vs. noisy input.
- Noise robustness: evaluate at σ ∈ {0.05, 0.10, 0.20, 0.30, 0.50}.
- Save all plots as high-resolution PNG to `results/`.
- Print comparison table: MLP vs RNN vs LSTM.

### FR-8: CLI Entry Point
- `uv run python src/main.py --mode [generate|train|evaluate|all|ui]`
- `--model [mlp|rnn|lstm|all]`
- `--seed <int>` (default 42)
- `--port <int>` (default 8050, only used with `--mode ui`)

### FR-9: Sinusoid Explorer — Interactive Dashboard
A browser-based interactive dashboard (Plotly Dash) titled **"SINUSOID EXPLORER"** with the following panels:

**9.1 Global Parameters Control Panel (left sidebar)**
- **Fs** slider: sampling frequency, range 10–2000 Hz.
- **N-cycles** integer control: number of displayed cycles, range 1–20.
- **BW** slider: bandwidth in Hz (used by the bandpass filter per component), range 0.1–100 Hz.
- **Display** toggle: `LINE` | `DOTS` — switches plot trace style globally.
- **Noise** dropdown: `None` | `Gaussian` | `Uniform` — noise model applied to combined signal.
- **Filter** dropdown: `None` | `Lowpass` | `Highpass` | `Bandpass` — filter applied per component with BPF checkbox.
- **SWEEP NOISE** button: animated sweep of noise σ from 0 → 1 → 0 to demonstrate robustness.

**9.2 Per-Sinusoid Controls (up to 4 sinusoids, color-coded)**
Each sinusoid block (Sin 1..Sin 4) contains:
- **MIX** checkbox: include/exclude component from the combined signal.
- **BPF** checkbox: apply per-component bandpass filter centred at *f* with width BW.
- **f** slider: frequency, range 0.1 Hz to Nyquist (Fs/2).
- **φ** slider: phase offset, range 0–2π rad.
- **A** slider: amplitude, range 0–2.0.

**9.3 Live Header Metrics Bar**
Displays computed metrics dynamically: **Fs** (Hz), **N-CYC**, **T** (total duration in seconds), **h** (total sample count), **F_MIN** (frequency resolution = Fs / h).

### FR-10: T-SNE 3D Visualization Tab
- 3-D interactive scatter plot of sliding-window features reduced to 3 dimensions via t-SNE.
- Points color-coded by frequency class label.
- Plotly 3D scene with rotate/pan/zoom; axis labels shown.

### FR-11: PCA 3D Visualization Tab
- 3-D interactive scatter plot of windowed features reduced via PCA to 3 principal components.
- Explained variance ratio (%) annotated per axis.
- Points color-coded by frequency class label.

### FR-12: FFT Spectrum Visualization Tab
- Magnitude FFT spectrum of the current combined signal.
- X-axis: frequency (Hz) up to Nyquist; optional log scale toggle.
- Vertical markers at each active sinusoid's frequency.
- BW bandwidth markers (shaded region) shown when BPF enabled.

### FR-13: UI Launch via CLI
- `uv run python src/main.py --mode ui` starts the Dash development server on `http://localhost:8050`.
- Ctrl+C exits cleanly.

---

## 8. Non-Functional Requirements

| ID    | Category       | Requirement                                             |
|-------|----------------|---------------------------------------------------------|
| NFR-1 | Performance    | Training of one model ≤ 10 min on CPU                  |
| NFR-2 | Reliability    | Deterministic results with fixed seed                   |
| NFR-3 | Maintainability| Every Python source file ≤ 145 code lines (v1.10; was 150) |
| NFR-4 | Security       | No secrets or API keys in source                        |
| NFR-5 | Testability    | ≥ 85 % branch coverage                                 |
| NFR-6 | Portability    | Python 3.10+, cross-platform (Linux/macOS/Windows)      |
| NFR-7 | Usability      | Single command reproduces all results                   |

---

## 9. User Stories

| ID   | Story                                                                                          |
|------|-----------------------------------------------------------------------------------------------|
| US-1 | As a grader, I want to run `uv sync && uv run python src/main.py --mode all` and see results.|
| US-2 | As a researcher, I want to change frequencies and noise levels via config without code edits. |
| US-3 | As an ML student, I want clear docstrings explaining each architecture choice.                |
| US-4 | As a reviewer, I want tests to pass and coverage to be ≥ 85 % automatically.                |
| US-5 | As a student, I want to open `--mode ui` and interactively explore sinusoid composition, noise effects, and FFT spectrum in a browser dashboard.      |

---

## 10. Assumptions

- Pure PyTorch is used (no high-level wrappers like Lightning).
- Dataset fits in RAM (< 1 GB).
- Training runs on CPU; GPU acceleration is a bonus if available.
- Amplitude A = 1.0 per sinusoid unless overridden via the UI sliders or config.
- The Sinusoid Explorer UI runs locally (Dash dev server); no production deployment is required.

---

## 11. Dependencies

| Dependency       | Purpose                                        | Version |
|------------------|------------------------------------------------|---------|
| torch            | Neural network training                        | ≥2.2    |
| numpy            | Signal generation, math                        | ≥1.26   |
| matplotlib       | Static plotting and PNG export                 | ≥3.8    |
| scipy            | FFT verification, bandpass filter design       | ≥1.12   |
| plotly           | Interactive charts (Signals, FFT, 3D scatter)  | ≥5.20   |
| dash             | Sinusoid Explorer browser dashboard (FR-9..13) | ≥2.16   |
| scikit-learn     | t-SNE and PCA dimensionality reduction         | ≥1.4    |
| pytest           | Testing framework                              | ≥8.0    |
| pytest-cov       | Coverage measurement                           | ≥4.1    |
| ruff             | Linting                                        | ≥0.4    |

---

## 12. Constraints

- No external paid APIs.
- No internet access required at runtime.
- Must use `uv` for all package management (no pip/venv).
- All config in `config/*.json`, no hardcoded operational values.

---

## 13. Out of Scope

- Transformer / attention-based models.
- Real-world (non-synthetic) signal data.
- GPU cluster or distributed training.
- Production deployment / containerisation of the Dash app (local dev server only).
- Audio processing (WAV, MP3 files).
- Mobile or native desktop app (Electron, PyQt) — browser-based Dash is sufficient.

---

## 14. Timeline and Milestones

| Phase | Milestone                              | Estimated Effort |
|-------|----------------------------------------|------------------|
| 1     | Documentation complete                 | 2 hours          |
| 2     | Project scaffold + shared infra        | 1 hour           |
| 3     | Data + model implementation            | 3 hours          |
| 4     | Training + evaluation                  | 2 hours          |
| 4F    | Sinusoid Explorer UI (FR-9..FR-13)     | 3 hours          |
| 5     | Tests + linting                        | 2 hours          |
| 6     | Experiments + README                   | 2 hours          |

---

## 15. Deliverables

1. GitHub repository with full source, tests, docs, results.
2. `README.md` as a complete lab report.
3. All result plots in `results/` (PNG, ≥ 150 dpi).
4. Passing test suite with ≥ 85 % coverage.
5. Zero Ruff violations.
6. Sinusoid Explorer dashboard accessible via `uv run python src/main.py --mode ui`.

---

## 16. Bibliography

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
2. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.
3. Shannon, C. E. (1949). Communication in the Presence of Noise. Proceedings of the IRE, 37(1), 10-21.
4. Nyquist, H. (1928). Certain Topics in Telegraph Transmission Theory. Transactions of the American Institute of Electrical Engineers, 47(2), 617-644.
