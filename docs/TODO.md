# TODO — HW1: Signal Frequency Extraction using Neural Networks

| Field    | Value                                                            |
|----------|------------------------------------------------------------------|
| Version  | 1.10                                                             |
| Updated  | 2026-04-28                                                       |
| Notes    | Decomposed into ≥ 75 atomic sub-tasks; max source-file length: **145 code lines** (was 150 in v1.00) |

Legend: 🔴 High | 🟡 Medium | 🟢 Low | ✅ Done | 🔄 In Progress | ⬜ Not Started

**Cross-reference notation**
- `PRD §X`     — section X of `docs/PRD.md`
- `PLAN §X`    — section X of `docs/PLAN.md`
- `PRD_sig`    — `docs/PRD_signal_generation.md`
- `PRD_mod`    — `docs/PRD_models.md`
- `PRD_tr`     — `docs/PRD_training.md`

**Constraint update (v1.10):** maximum Python source-file length reduced from
150 → **145 code lines** (excluding blank lines and comment-only lines). This is
enforced by NFR-3 and audit task 8.7. Tests files obey the same limit.

---

## Phase 1 — Documentation

| #    | Task                                          | Pri | Status | Refs               | Definition of Done                                              |
|------|-----------------------------------------------|-----|--------|--------------------|-----------------------------------------------------------------|
| 1.1  | Create docs/PRD.md with all 15 sections       | 🔴  | ✅     | PRD all            | Sections 1–15 present; KPI table; FR-1..FR-8; NFR-1..NFR-7      |
| 1.2  | Update PRD NFR-3 + KPI to 145-line limit       | 🔴  | ✅     | PRD §5, §8         | NFR-3 reads "≤ 145 code lines"; KPI row updated                 |
| 1.3  | Create docs/PLAN.md with C4 + ADRs            | 🔴  | ✅     | PLAN all           | Context, Container, Component diagrams; 5 ADRs                  |
| 1.4  | Create docs/PRD_signal_generation.md          | 🔴  | ✅     | PRD FR-1, FR-2     | Inputs, outputs, 7 SG-T* test scenarios documented              |
| 1.5  | Create docs/PRD_models.md                     | 🔴  | ✅     | PRD FR-3..FR-5     | All 3 architectures with parameter counts and rationale         |
| 1.6  | Create docs/PRD_training.md                   | 🔴  | ✅     | PRD FR-6           | Loss, optimizer, schedule, early stop, 7 TR-T* scenarios        |
| 1.7  | Create docs/PROMPTS.md                        | 🟡  | ✅     | All                | All AI prompts logged with purpose, output, lessons             |
| 1.8  | Decompose docs/TODO.md to ≥ 300 lines         | 🔴  | ✅     | All                | Atomic sub-tasks; cross-refs to PRD/PLAN                        |
| 1.9  | Document gatekeeper extension points          | 🟡  | ⬜     | PLAN §11           | Extension matrix updated with HTTP-adapter scaffold notes       |
| 1.10 | Add Bibliography section to PRD               | 🟢  | ⬜     | PRD §15            | Cite Hochreiter 1997, Adam paper, Nyquist                       |

---

## Phase 2 — Project Scaffolding

| #    | Task                                          | Pri | Status | Refs               | Definition of Done                                              |
|------|-----------------------------------------------|-----|--------|--------------------|-----------------------------------------------------------------|
| 2.1  | Create pyproject.toml with project metadata   | 🔴  | ✅     | PRD §11            | name, version, deps; `uv sync` resolves cleanly                 |
| 2.2  | Configure ruff in pyproject.toml              | 🔴  | ✅     | PRD KPI            | line-length=100, target=py310, rules E,F,W,I,N,UP,B,C4,SIM      |
| 2.3  | Configure pytest + coverage                   | 🔴  | ✅     | NFR-5              | fail_under=85; source=src; omit main.py                         |
| 2.4  | Create config/setup.json (v1.00)              | 🔴  | ✅     | PRD FR-1..FR-7     | All hyperparameters present                                     |
| 2.5  | Create config/rate_limits.json (v1.00)        | 🔴  | ✅     | PLAN §6            | default, file_io, checkpoint services defined                   |
| 2.6  | Create config/logging_config.json (v1.00)     | 🔴  | ✅     | PLAN §8            | Console + file handlers; standard + detailed formatters         |
| 2.7  | Create .env-example                           | 🔴  | ✅     | NFR-4              | Placeholders only; commented                                    |
| 2.8  | Create .gitignore                             | 🔴  | ⬜     | NFR-4              | .env, *.key, *.pem, secrets.*, __pycache__, .venv/              |
| 2.9  | Create directory tree                         | 🔴  | ✅     | PLAN §1            | src/, tests/, docs/, data/, results/, assets/, notebooks/       |
| 2.10 | Add LICENSE (MIT)                             | 🟡  | ⬜     | PRD §15            | MIT text with author + year                                     |
| 2.11 | Add CITATION.cff (optional)                   | 🟢  | ⬜     | PRD §15            | Machine-readable citation                                       |

---

## Phase 3 — Shared Infrastructure

| #    | Task                                          | Pri | Status | Refs               | Definition of Done                                              |
|------|-----------------------------------------------|-----|--------|--------------------|-----------------------------------------------------------------|
| 3.1  | Create shared/version.py with CODE_VERSION    | 🔴  | ✅     | PLAN §6            | CODE_VERSION="1.00"; parse_version(); validate_config_version() |
| 3.2  | Add MIN/MAX config-version constants          | 🔴  | ✅     | PLAN §6            | Out-of-range raises ValueError with message                     |
| 3.3  | Create shared/config.py with cached loader    | 🔴  | ✅     | PLAN §6            | get_setup(), get_rate_limits(), get_logging_config()            |
| 3.4  | Honour env-var overrides for config dir       | 🔴  | ✅     | NFR-4              | FREQ_EXTRACTOR_CONFIG_DIR respected                             |
| 3.5  | Add setup_logging() helper                    | 🔴  | ✅     | PLAN §8            | dictConfig from logging_config.json; ensures log dir exists     |
| 3.6  | Create shared/gatekeeper.py with ApiGatekeeper| 🔴  | ✅     | PLAN §6, ADR-4     | execute(), get_queue_status(), exponential-backoff retries      |
| 3.7  | Implement sliding-window rate limiter         | 🔴  | ✅     | PLAN §6            | 60-second window; sleeps when limit reached                     |
| 3.8  | Add singleton get_gatekeeper() factory        | 🔴  | ✅     | PLAN §6            | One instance per service name; cached                           |
| 3.9  | Create constants.py                           | 🔴  | ✅     | PLAN §6            | MODEL_TYPES, FREQUENCY_LABELS, SPLIT_NAMES, TENSOR_DTYPES       |
| 3.10 | Create __init__.py files in every package     | 🔴  | ✅     | NFR-6              | freq_extractor, sdk, services, shared all importable            |
| 3.11 | Define __all__ in package __init__.py         | 🟡  | ✅     | NFR-6              | Explicit public API surface                                     |
| 3.12 | Add module-level docstrings to all shared/*   | 🔴  | ✅     | NFR-3              | Every shared module has a docstring                             |
| 3.13 | Verify each shared file ≤ 145 code lines      | 🔴  | ✅     | NFR-3 (v1.10)      | Line counter confirms compliance                                |

---

## Phase 4 — Services Layer (Highly Decomposed)

### 4A — DataService (services/data_service.py)

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 4A.1  | Implement SignalGenerator class               | 🔴  | ✅     | PRD FR-1, PRD_sig §4       | generate_clean(), generate_noisy() return ndarray (N,)          |
| 4A.2  | Validate signal-gen inputs (range checks)     | 🔴  | ✅     | PRD_sig §11 SG-T6          | f>0, fs ≥ 2·f_max, dur>0, σ≥0; ValueError otherwise            |
| 4A.3  | Implement Nyquist guard at startup            | 🔴  | ✅     | PRD_sig §2                 | Raises if max(frequencies) > sampling_rate / 2                  |
| 4A.4  | Implement DatasetBuilder (sliding window)     | 🔴  | ✅     | PRD FR-2, PRD_sig §6       | Produces (N − W) entries per signal                             |
| 4A.5  | Implement one-hot encoder for freq labels     | 🔴  | ✅     | PRD FR-2                   | Shape (4,) float32; sums to 1.0                                 |
| 4A.6  | Implement DatasetSplitter (stratified 70/15/15)| 🔴 | ✅     | PRD FR-2                   | Each split contains entries for all 4 frequencies               |
| 4A.7  | Implement DataNormalizer (fit on train only)  | 🔴  | ✅     | PRD_sig §7                 | No leakage; same mean/std applied to val + test                 |
| 4A.8  | Implement DataPersistence via Gatekeeper      | 🔴  | ✅     | PLAN §6, ADR-4             | save/load .npz; all I/O via get_gatekeeper("file_io")           |
| 4A.9  | Build PyTorch Dataset for MLP                 | 🔴  | ✅     | PLAN §5                    | __getitem__ returns (X[14], y[1])                               |
| 4A.10 | Build PyTorch Dataset for sequential models   | 🔴  | ✅     | PLAN §5                    | __getitem__ returns (X[10,5], y[1])                             |
| 4A.11 | Add DataLoader factory with worker_init_fn    | 🔴  | ✅     | PRD_tr §8                  | Determinism: same seed → same batches                           |
| 4A.12 | Verify file ≤ 145 code lines                  | 🔴  | ✅     | NFR-3 (v1.10)              | data_service.py within new limit; split if needed               |

### 4B — Model Definitions (one file per architecture)

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 4B.1  | Implement MLPModel (services/mlp_model.py)    | 🔴  | ✅     | PRD FR-3, PRD_mod §3       | 14→64→128→64→1 with Tanh; forward returns (B,1)                 |
| 4B.2  | Implement RNNModel (services/rnn_model.py)    | 🔴  | ✅     | PRD FR-4, PRD_mod §4       | 2-layer RNN, hidden=64; final hidden → Linear → (B,1)           |
| 4B.3  | Implement LSTMModel (services/lstm_model.py)  | 🔴  | ✅     | PRD FR-5, PRD_mod §5       | 2-layer LSTM, hidden=64; final hidden → Linear → (B,1)          |
| 4B.4  | Add model.count_parameters() helper           | 🟡  | ✅     | PRD_mod §6                 | Returns trainable parameter count                               |
| 4B.5  | Implement ModelFactory.create_model(type)     | 🔴  | ✅     | PLAN §6                    | "mlp" \| "rnn" \| "lstm" → correct class; raises on unknown     |
| 4B.6  | Verify orthogonal init for RNN/LSTM weights   | 🟡  | ✅     | PRD_mod §7                 | Recurrent weights initialised orthogonally                      |
| 4B.7  | Confirm dropout=0.1 applied between layers    | 🟡  | ✅     | PRD FR-4, FR-5             | Both RNN and LSTM use dropout between layers                    |
| 4B.8  | Verify each model file ≤ 145 code lines       | 🔴  | ✅     | NFR-3 (v1.10)              | mlp_model.py, rnn_model.py, lstm_model.py all comply            |

### 4C — TrainingService (services/training_service.py)

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 4C.1  | Implement train_one_epoch()                   | 🔴  | ✅     | PRD FR-6, PRD_tr §7        | Returns mean training loss for the epoch                        |
| 4C.2  | Implement evaluate() (val/test loop)          | 🔴  | ✅     | PRD FR-6                   | Returns mean MSE; eval() mode; no_grad context                  |
| 4C.3  | Implement EarlyStopping helper class          | 🔴  | ✅     | PRD_tr §6                  | patience, min_delta; restores best weights                      |
| 4C.4  | Add ReduceLROnPlateau scheduler hook          | 🔴  | ✅     | PRD_tr §4                  | Halves LR after lr_reduce_patience epochs of no improvement     |
| 4C.5  | Add gradient clipping (max_norm=1.0)          | 🔴  | ✅     | PRD_tr §7                  | Applied before optimizer.step() in every iteration              |
| 4C.6  | Add checkpoint save via Gatekeeper            | 🔴  | ✅     | PLAN §5, ADR-4             | Schema matches PLAN §5; saved to results/checkpoints/           |
| 4C.7  | Add checkpoint load + state restoration       | 🔴  | ✅     | PLAN §5                    | Validates config_version; restores model + optimizer            |
| 4C.8  | Set all 4 random seeds at training start      | 🔴  | ✅     | PRD_tr §8                  | torch + numpy + random + cudnn all seeded                       |
| 4C.9  | Log per-epoch train/val MSE                   | 🟡  | ✅     | PRD FR-6                   | Structured log lines (one per epoch)                            |
| 4C.10 | Verify training_service.py ≤ 145 code lines   | 🔴  | ✅     | NFR-3 (v1.10)              | Split into helper modules if over limit                         |

### 4D — EvaluationService (services/evaluation_service.py)

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 4D.1  | Implement compute_split_mse()                 | 🔴  | ✅     | PRD FR-7                   | Returns dict {train, val, test} → MSE                           |
| 4D.2  | Implement plot_training_curves()              | 🔴  | ✅     | PRD_tr §11                 | Saves results/training_curves.png at ≥150 dpi                   |
| 4D.3  | Implement plot_predictions() per model        | 🔴  | ✅     | PRD_tr §11                 | Saves results/predictions_<model>.png for each                  |
| 4D.4  | Implement plot_noise_robustness()             | 🔴  | ✅     | PRD FR-7, PRD_tr §10       | Saves results/noise_robustness.png; tests σ ∈ {.05..0.50}       |
| 4D.5  | Implement plot_per_frequency_mse()            | 🔴  | ✅     | PRD FR-7                   | Bar chart: 3 models × 4 frequencies                             |
| 4D.6  | Implement plot_signal_examples()              | 🔴  | ✅     | PRD FR-7                   | Clean / noisy / target overlay per frequency                    |
| 4D.7  | Implement build_comparison_table()            | 🔴  | ✅     | PRD FR-7                   | Markdown table with train/val/test MSE per model                |
| 4D.8  | Apply consistent plot style (axes, fonts)     | 🟡  | ✅     | PRD FR-7                   | All plots: title, axis labels, legend, accessible colours       |
| 4D.9  | Verify evaluation_service.py ≤ 145 code lines | 🔴  | ✅     | NFR-3 (v1.10)              | Split into plot_helpers.py if necessary                         |

### 4E — Cross-Cutting Service Concerns

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 4E.1  | Verify ALL service files ≤ 145 code lines     | 🔴  | ✅     | NFR-3 (v1.10)              | Line-counter script confirms compliance                         |
| 4E.2  | Add docstring to every public class/function  | 🔴  | ✅     | NFR-3                      | Manual review or ruff D-rules                                   |
| 4E.3  | Confirm no service imports sdk.py             | 🔴  | ✅     | PLAN §1                    | Static grep returns 0 hits                                      |
| 4E.4  | Confirm no service hardcodes operational vals | 🔴  | ✅     | NFR-4                      | Frequencies, paths, etc. all loaded from config                 |

### 4F — UIService: Sinusoid Explorer Dashboard (services/ui_service.py)

#### 4F-1 — Scaffold & Layout

| #      | Task                                                       | Pri | Status | Refs                        | Definition of Done                                                    |
|--------|------------------------------------------------------------|-----|--------|-----------------------------|-----------------------------------------------------------------------|
| 4F-1.1 | Create services/ui_service.py skeleton                     | 🔴  | ✅     | PRD FR-9, PLAN ADR-6        | `UIService` class; `build_app()` returns Dash app object              |
| 4F-1.2 | Apply dark theme (background #0d1117, text #e6edf3)        | 🔴  | ✅     | PRD FR-9                    | All panels match screenshots' dark palette                            |
| 4F-1.3 | Build header metrics bar (Fs, N-CYC, T, h, F_MIN)         | 🔴  | ✅     | PRD FR-9 §9.3               | Bar auto-updates on any parameter change                              |
| 4F-1.4 | Create left sidebar: Global Parameters panel               | 🔴  | ✅     | PRD FR-9 §9.1               | Fs slider, N-cycles, BW, Display toggle, Noise dropdown, Filter DD    |
| 4F-1.5 | Add SWEEP NOISE animated button                            | 🔴  | ✅     | PRD FR-9 §9.1               | Interval component sweeps σ 0→1→0; updates combined signal plot       |
| 4F-1.6 | Create per-sinusoid control blocks (Sin 1..Sin 4)          | 🔴  | ✅     | PRD FR-9 §9.2               | Each has MIX/BPF checkboxes, f/φ/A sliders, unique colour dot        |
| 4F-1.7 | Verify ui_service.py split across helper files ≤ 145 lines | 🔴  | ✅     | NFR-3 (v1.10)               | ui_service.py + ui_callbacks.py + ui_layout.py each ≤ 145 lines      |

#### 4F-2 — SIGNALS Tab

| #      | Task                                                       | Pri | Status | Refs                        | Definition of Done                                                    |
|--------|------------------------------------------------------------|-----|--------|-----------------------------|-----------------------------------------------------------------------|
| 4F-2.1 | Implement Individual Sinusoids upper plot                  | 🔴  | ✅     | PRD FR-9 §9.4               | One Plotly trace per active sinusoid; colour matches Sin N dot        |
| 4F-2.2 | Implement Combined Signal — Clean lower plot               | 🔴  | ✅     | PRD FR-9 §9.4               | Sum of MIX-checked components; white trace; label "Mixed clean"       |
| 4F-2.3 | Support LINE / DOTS display toggle                         | 🔴  | ✅     | PRD FR-9 §9.1               | Toggling changes `mode` in all traces to "lines" or "markers"         |
| 4F-2.4 | Hover tooltip shows (t, value) on both plots               | 🔴  | ✅     | PRD FR-9 §9.4               | Plotly hovertemplate set; verified on both plots                      |
| 4F-2.5 | Plot toolbar: camera (save PNG), zoom, reset axes          | 🔴  | ✅     | PRD FR-9 §9.4               | Plotly config modebar_add includes camera + zoom                      |
| 4F-2.6 | Apply per-sinusoid bandpass filter when BPF checked        | 🔴  | ✅     | PRD FR-9 §9.2               | scipy.signal.butter bandpass centred at f ± BW/2                      |
| 4F-2.7 | Apply global noise to combined signal based on Noise DD    | 🟡  | ✅     | PRD FR-9 §9.1               | "Gaussian" adds N(0, σ²); "Uniform" adds U(−σ, σ); "None" = clean    |

#### 4F-3 — T-SNE 3D Tab

| #      | Task                                                       | Pri | Status | Refs                        | Definition of Done                                                    |
|--------|------------------------------------------------------------|-----|--------|-----------------------------|-----------------------------------------------------------------------|
| 4F-3.1 | Compute sliding-window features from active sinusoids      | 🔴  | ✅     | PRD FR-10                   | Feature matrix shape (N_windows, window_size)                         |
| 4F-3.2 | Run sklearn TSNE(n_components=3) on features               | 🔴  | ✅     | PRD FR-10                   | Result shape (N_windows, 3)                                           |
| 4F-3.3 | Render Plotly go.Scatter3d colour-coded by frequency label | 🔴  | ✅     | PRD FR-10                   | 4 colours match Sin 1..4; interactive rotate/zoom                     |
| 4F-3.4 | Show perplexity and iteration controls in tab header       | 🟡  | ✅     | PRD FR-10                   | Sliders update t-SNE on change                                        |

#### 4F-4 — PCA 3D Tab

| #      | Task                                                       | Pri | Status | Refs                        | Definition of Done                                                    |
|--------|------------------------------------------------------------|-----|--------|-----------------------------|-----------------------------------------------------------------------|
| 4F-4.1 | Run sklearn PCA(n_components=3) on same feature matrix     | 🔴  | ✅     | PRD FR-11                   | Result shape (N_windows, 3)                                           |
| 4F-4.2 | Render Plotly go.Scatter3d colour-coded by frequency label | 🔴  | ✅     | PRD FR-11                   | 4 colours; interactive rotate/zoom                                    |
| 4F-4.3 | Annotate each axis with explained variance ratio (%)       | 🔴  | ✅     | PRD FR-11                   | Axis title = "PC1 (42.3 %)" format                                    |

#### 4F-5 — FFT Spectrum Tab

| #      | Task                                                       | Pri | Status | Refs                        | Definition of Done                                                    |
|--------|------------------------------------------------------------|-----|--------|-----------------------------|-----------------------------------------------------------------------|
| 4F-5.1 | Compute np.fft.rfft on combined clean signal               | 🔴  | ✅     | PRD FR-12                   | Magnitude spectrum computed correctly                                 |
| 4F-5.2 | Render Plotly line chart: frequency (Hz) vs magnitude      | 🔴  | ✅     | PRD FR-12                   | X-axis 0..Nyquist; Y-axis magnitude                                   |
| 4F-5.3 | Add vertical markers at each active sinusoid's frequency   | 🔴  | ✅     | PRD FR-12                   | Colour-matched dashed vertical lines                                  |
| 4F-5.4 | Add BW shaded regions when BPF enabled                     | 🟡  | ✅     | PRD FR-12                   | go.Scatter fill between f−BW/2 and f+BW/2                            |
| 4F-5.5 | Add log-scale toggle for frequency axis                    | 🟡  | ✅     | PRD FR-12                   | Button switches xaxis.type log ↔ linear                               |

#### 4F-6 — CLI Integration & Validation

| #      | Task                                                       | Pri | Status | Refs                        | Definition of Done                                                    |
|--------|------------------------------------------------------------|-----|--------|-----------------------------|-----------------------------------------------------------------------|
| 4F-6.1 | Add `launch_ui(port)` method to UIService                  | 🔴  | ✅     | PRD FR-13                   | Calls `app.run(debug=False, port=port)`                               |
| 4F-6.2 | Add `--mode ui` to SDK and main.py                         | 🔴  | ⬜     | PRD FR-13, FR-8             | Calls `sdk.launch_ui(port=args.port)`                                 |
| 4F-6.3 | Add `--port` argument to main.py (default 8050)            | 🟡  | ⬜     | PRD FR-13                   | Argparse integer; passed through to UIService                         |
| 4F-6.4 | Verify Dash app starts and responds on port 8050           | 🔴  | ⬜     | PRD FR-13                   | `uv run python src/main.py --mode ui` opens browser; 200 OK response  |

---

## Phase 5 — SDK + Entry Point

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 5.1   | Create sdk/__init__.py exposing SDK class     | 🔴  | ⬜     | PLAN §1                    | `from freq_extractor.sdk import FreqExtractorSDK` works         |
| 5.2   | Implement FreqExtractorSDK.generate_data()    | 🔴  | ⬜     | PRD FR-1, FR-2             | Wraps DataService; returns split datasets                       |
| 5.3   | Implement FreqExtractorSDK.train(model_type)  | 🔴  | ⬜     | PRD FR-6                   | Wraps TrainingService; returns checkpoint path + final mse      |
| 5.4   | Implement FreqExtractorSDK.evaluate(model)    | 🔴  | ⬜     | PRD FR-7                   | Wraps EvaluationService; returns metrics dict                   |
| 5.5   | Implement FreqExtractorSDK.run_all()          | 🔴  | ⬜     | PRD US-1, FR-8             | Generate → train all 3 → evaluate → save all plots              |
| 5.6   | Verify CLI does NOT import services directly  | 🔴  | ⬜     | PLAN §1                    | Only imports sdk.* — static grep                                |
| 5.7   | Create src/main.py CLI                        | 🔴  | ⬜     | PRD FR-8                   | --mode, --model, --seed args parsed                             |
| 5.8   | Add CLI argument validation                   | 🔴  | ⬜     | PRD FR-8                   | Invalid choices → user-friendly argparse error                  |
| 5.9   | Honour FREQ_EXTRACTOR_FORCE_CPU env var       | 🟡  | ⬜     | .env-example               | Forces CPU when set to "1"                                      |
| 5.10  | Verify sdk.py + main.py ≤ 145 code lines      | 🔴  | ⬜     | NFR-3 (v1.10)              | Within new limit                                                |

---

## Phase 6 — Tests (Heavily Decomposed)

### 6A — Test Infrastructure

| #     | Task                                          | Pri | Status | Refs               | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|--------------------|-----------------------------------------------------------------|
| 6A.1  | Create tests/conftest.py with shared fixtures | 🔴  | ⬜     | PLAN §10           | Fixtures: tmp_config_dir, sample_signal, small_dataset          |
| 6A.2  | Add `set_seed` autouse fixture                | 🔴  | ⬜     | PRD_tr §8          | All tests deterministic                                         |
| 6A.3  | Add tmp_path-based gatekeeper fixture         | 🔴  | ⬜     | PLAN §10           | Gatekeeper isolated per test                                    |
| 6A.4  | Add small_dataset_factory fixture             | 🔴  | ⬜     | PLAN §10           | 80-entry dataset for fast tests                                 |

### 6B — Signal-Generation Tests (test_data_service.py)

| #     | Test ID         | Scenario                                    | Status | Refs            | Definition of Done                              |
|-------|-----------------|---------------------------------------------|--------|-----------------|-------------------------------------------------|
| 6B.1  | SG-T1           | FFT peak matches configured frequency       | ⬜     | PRD_sig §11     | abs(peak_f − expected_f) < 0.5 Hz               |
| 6B.2  | SG-T2           | Noisy SNR ≈ 20 dB at σ=0.1                  | ⬜     | PRD_sig §11     | Computed SNR within ±2 dB                       |
| 6B.3  | SG-T3           | Sliding-window count = N − W                | ⬜     | PRD_sig §11     | 1990 entries per 2000-sample signal             |
| 6B.4  | SG-T4           | Reproducibility same seed → identical data  | ⬜     | PRD_sig §11     | np.array_equal across two runs                  |
| 6B.5  | SG-T5           | σ=0 → noisy == clean                        | ⬜     | PRD_sig §11     | Element-wise equality                           |
| 6B.6  | SG-T6           | window_size > N raises ValueError           | ⬜     | PRD_sig §11     | pytest.raises(ValueError)                       |
| 6B.7  | SG-T7           | Splits sum to total                         | ⬜     | PRD_sig §11     | len(train)+len(val)+len(test) == total          |
| 6B.8  | SG-T8 (new)     | One-hot label correctness                   | ⬜     | PRD FR-2        | sum(one_hot)==1; argmax matches freq index      |
| 6B.9  | SG-T9 (new)     | Stratified split distributes all 4 freqs    | ⬜     | PRD FR-2        | Each split contains entries for every freq      |
| 6B.10 | SG-T10 (new)    | Normaliser fit-on-train (no leakage)        | ⬜     | PRD_sig §7      | val/test stats ≠ train stats unless coincid.    |
| 6B.11 | SG-T11 (new)    | Negative noise std raises ValueError        | ⬜     | PRD_sig §11     | pytest.raises(ValueError)                       |
| 6B.12 | SG-T12 (new)    | Nyquist violation raises ValueError         | ⬜     | PRD_sig §2      | fs < 2·f_max → ValueError                       |
| 6B.13 | SG-T13 (new)    | Zero-length duration raises ValueError      | ⬜     | PRD_sig §11     | duration_s ≤ 0 → ValueError                     |
| 6B.14 | SG-T14 (new)    | Noise mean ≈ 0 (statistical test)           | ⬜     | PRD FR-1        | abs(mean(noise)) < 3·σ/√N                       |
| 6B.15 | SG-T15 (new)    | Noise std matches σ·A within 5 %            | ⬜     | PRD FR-1        | abs(std(noise) − σA)/σA < 0.05                  |

### 6C — MLP-Specific Tests (test_models.py::TestMLP)

| #     | Test ID         | Scenario                                    | Status | Refs            | Definition of Done                              |
|-------|-----------------|---------------------------------------------|--------|-----------------|-------------------------------------------------|
| 6C.1  | MLP-T1          | Forward pass shape (B=32) → (32,1)          | ⬜     | PRD_mod §3      | Output shape correct; no NaN                    |
| 6C.2  | MLP-T2          | Forward pass batch=1 works                  | ⬜     | PRD_mod §9      | No crash; (1,1) output                          |
| 6C.3  | MLP-T3          | Parameter count ≈ 17 601 (±5 %)             | ⬜     | PRD_mod §6      | count_parameters() within tolerance              |
| 6C.4  | MLP-T4          | Gradients flow through all 4 linear layers  | ⬜     | PRD_mod §9      | Every param.grad is not None and != 0           |
| 6C.5  | MLP-T5          | Tanh activation present                     | ⬜     | PRD FR-3        | nn.Tanh found in module list                    |
| 6C.6  | MLP-T6          | Wrong input dim raises clear error          | ⬜     | PRD_mod §9      | Input shape (B,13) → RuntimeError               |
| 6C.7  | MLP-T7          | Serialise + deserialise via torch.save      | ⬜     | PLAN §5         | Loaded model gives identical output             |

### 6D — RNN-Specific Tests (test_models.py::TestRNN)

| #     | Test ID         | Scenario                                    | Status | Refs            | Definition of Done                              |
|-------|-----------------|---------------------------------------------|--------|-----------------|-------------------------------------------------|
| 6D.1  | RNN-T1          | Forward pass shape (B=32, T=10) → (32,1)    | ⬜     | PRD_mod §4      | Output shape correct; no NaN                    |
| 6D.2  | RNN-T2          | hidden_size == 64                           | ⬜     | PRD FR-4        | Internal hidden tensor has size 64              |
| 6D.3  | RNN-T3          | num_layers == 2                             | ⬜     | PRD FR-4        | self.rnn.num_layers == 2                        |
| 6D.4  | RNN-T4          | Variable sequence length supported          | ⬜     | PRD_mod §9      | T=5 and T=20 both run without error             |
| 6D.5  | RNN-T5          | Recurrent weights orthogonally initialised  | ⬜     | PRD_mod §7      | Singular values ≈ 1                             |
| 6D.6  | RNN-T6          | Gradients flow through both layers          | ⬜     | PRD_mod §9      | All recurrent params have non-zero grad         |
| 6D.7  | RNN-T7          | Dropout active in train(), inactive in eval | ⬜     | PRD FR-4        | Different outputs in train; identical in eval   |

### 6E — LSTM-Specific Tests (test_models.py::TestLSTM)

| #     | Test ID         | Scenario                                    | Status | Refs            | Definition of Done                              |
|-------|-----------------|---------------------------------------------|--------|-----------------|-------------------------------------------------|
| 6E.1  | LSTM-T1         | Forward pass shape (B=32, T=10) → (32,1)    | ⬜     | PRD_mod §5      | Output shape correct; no NaN                    |
| 6E.2  | LSTM-T2         | Cell + hidden state both initialised        | ⬜     | PRD_mod §5      | h₀ and c₀ shapes (num_layers, B, 64)            |
| 6E.3  | LSTM-T3         | Parameter count > RNN parameter count       | ⬜     | PRD_mod §6      | params(LSTM) ≈ 4 × params(RNN)                  |
| 6E.4  | LSTM-T4         | Forget gate biases present                  | ⬜     | PRD_mod §5      | bias_hh_l0 has 4·hidden_size entries            |
| 6E.5  | LSTM-T5         | Gradients flow through all 4 gate matrices  | ⬜     | PRD_mod §9      | weight_ih_l0 has 4·hidden rows w/ grads         |
| 6E.6  | LSTM-T6         | Eval mode disables dropout                  | ⬜     | PRD FR-5        | Identical outputs in repeated eval calls        |
| 6E.7  | LSTM-T7         | Outperforms MLP on f₁=5 Hz on smoke set     | 🟡 ⬜  | PRD_mod §5      | LSTM val MSE < MLP val MSE on f₁ subset         |

### 6F — Model-Factory Tests

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 6F.1  | Factory returns MLPModel for "mlp"            | 🔴  | ⬜     | PRD_mod §9 MD-T6           | isinstance check passes                                         |
| 6F.2  | Factory returns RNNModel for "rnn"            | 🔴  | ⬜     | PRD_mod §9                 | isinstance check passes                                         |
| 6F.3  | Factory returns LSTMModel for "lstm"          | 🔴  | ⬜     | PRD_mod §9                 | isinstance check passes                                         |
| 6F.4  | Factory raises on unknown model type          | 🔴  | ⬜     | PRD_mod §9                 | "transformer" → ValueError                                      |

### 6G — Training-Service Tests

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 6G.1  | TR-T1 smoke: 2-epoch train decreases loss     | 🔴  | ⬜     | PRD_tr §12                 | Final loss < initial loss                                       |
| 6G.2  | TR-T2 early stopping fires at patience=2      | 🔴  | ⬜     | PRD_tr §12                 | Stops within ≤ patience+1 epochs                                |
| 6G.3  | TR-T3 grad clipping prevents NaN              | 🔴  | ⬜     | PRD_tr §12                 | All losses finite under exploding gradients                     |
| 6G.4  | TR-T4 checkpoint save + load round-trip       | 🔴  | ⬜     | PRD_tr §12                 | Loaded model gives identical predictions                        |
| 6G.5  | TR-T5 LR scheduler halves LR on plateau       | 🔴  | ⬜     | PRD_tr §12                 | optimizer.lr halved after patience epochs                       |
| 6G.6  | TR-T6 MSE(y, y) == 0                          | 🔴  | ⬜     | PRD_tr §12                 | Trivial equality test                                           |
| 6G.7  | TR-T7 reproducibility two runs same seed      | 🔴  | ⬜     | PRD_tr §12                 | Identical final val_mse across runs                             |
| 6G.8  | Training raises on empty DataLoader           | 🟡  | ⬜     | Edge case                  | Clear error if dataset is empty                                 |

---

### 6H — Shared-Infrastructure Tests (test_shared.py)

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 6H.1  | parse_version("1.00") → (1,0)                 | 🔴  | ⬜     | PLAN §6                    | Tuple equality                                                  |
| 6H.2  | parse_version invalid raises ValueError       | 🔴  | ⬜     | PLAN §6                    | "abc" → ValueError                                              |
| 6H.3  | Major version mismatch rejected               | 🔴  | ⬜     | PLAN §6                    | "2.00" config → ValueError                                      |
| 6H.4  | Out-of-range minor rejected                   | 🔴  | ⬜     | PLAN §6                    | Above MAX_CONFIG_VERSION → ValueError                           |
| 6H.5  | Config cache returns same object              | 🟡  | ⬜     | PLAN §6                    | id(get_setup()) == id(get_setup())                              |
| 6H.6  | clear_cache() forces re-read                  | 🟡  | ⬜     | PLAN §6                    | New file content seen after clear_cache()                       |
| 6H.7  | Missing config file raises FileNotFoundError  | 🔴  | ⬜     | PLAN §8                    | Clear error message                                             |
| 6H.8  | Gatekeeper executes successful call           | 🔴  | ⬜     | PLAN §6                    | Return value forwarded; total_calls += 1                        |
| 6H.9  | Gatekeeper retries on transient failure       | 🔴  | ⬜     | PLAN §6                    | Eventually succeeds; total_retries > 0                          |
| 6H.10 | Gatekeeper raises after max_retries           | 🔴  | ⬜     | PLAN §6                    | GatekeeperError raised                                          |
| 6H.11 | get_queue_status returns 4 keys               | 🔴  | ⬜     | PLAN §6                    | queue_depth, total_calls, errors, retries                       |
| 6H.12 | get_gatekeeper singleton per service name     | 🔴  | ⬜     | PLAN §6                    | Same instance returned twice                                    |
| 6H.13 | Sliding-window rate limiter sleeps when full  | 🟡  | ⬜     | PLAN §6                    | Mocked time.sleep called when limit reached                     |

### 6I — Integration Tests (test_pipeline.py)

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 6I.1  | End-to-end run_all() with tiny config         | 🔴  | ⬜     | PRD US-1                   | All 3 models train and produce metrics                          |
| 6I.2  | Saved checkpoints load + reproduce metrics    | 🔴  | ⬜     | PRD_tr §13                 | Identical test MSE on reload                                    |
| 6I.3  | All required result files present             | 🔴  | ⬜     | PRD_tr §11                 | 7 PNG files present in results/                                 |
| 6I.4  | CLI invocation works (subprocess)             | 🟡  | ⬜     | PRD FR-8                   | exit code 0; stdout contains "complete"                         |
| 6I.5  | CLI rejects invalid model type                | 🟡  | ⬜     | PRD FR-8                   | exit code != 0; informative stderr                              |

### 6J — Coverage

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 6J.1  | Reach ≥ 85 % coverage globally                | 🔴  | ⬜     | NFR-5                      | `pytest --cov=src` reports ≥ 85 %                               |
| 6J.2  | Each shared module ≥ 90 %                     | 🔴  | ⬜     | PLAN §10                   | shared/* coverage ≥ 90 %                                        |
| 6J.3  | Each service module ≥ 90 %                    | 🔴  | ⬜     | PLAN §10                   | services/* coverage ≥ 90 %                                      |
| 6J.4  | Verify all test files ≤ 145 code lines        | 🔴  | ⬜     | NFR-3 (v1.10)              | Test files obey same length rule                                |

### 6K — UIService Tests (tests/unit/test_ui_service.py)

| #     | Task                                           | Pri | Status | Refs                       | Definition of Done                                                     |
|-------|------------------------------------------------|-----|--------|----------------------------|------------------------------------------------------------------------|
| 6K.1  | build_app() returns Dash app object            | 🔴  | ⬜     | PRD FR-9                   | isinstance(app, dash.Dash)                                             |
| 6K.2  | Header metrics update on Fs change             | 🔴  | ⬜     | PRD FR-9 §9.3              | Callback output contains updated Fs, T, h, F_MIN values                |
| 6K.3  | Individual Sinusoids trace count == active Mix | 🔴  | ⬜     | PRD FR-9 §9.4              | 2 checked MIX → 2 traces in upper figure                               |
| 6K.4  | Combined signal = sum of MIX components        | 🔴  | ⬜     | PRD FR-9 §9.4              | Mathematical equality within float tolerance                           |
| 6K.5  | DOTS display mode sets trace mode="markers"    | 🔴  | ⬜     | PRD FR-9 §9.1              | All traces mode=="markers" when DOTS selected                          |
| 6K.6  | LINE display mode sets trace mode="lines"      | 🔴  | ⬜     | PRD FR-9 §9.1              | All traces mode=="lines" when LINE selected                            |
| 6K.7  | BPF checkbox triggers bandpass filter          | 🔴  | ⬜     | PRD FR-9 §9.2              | Filtered signal has attenuation outside f ± BW/2                       |
| 6K.8  | Gaussian noise added to combined when selected | 🔴  | ⬜     | PRD FR-9 §9.1              | clean != noisy when noise=="Gaussian"                                  |
| 6K.9  | FFT tab: peak matches active sinusoid f        | 🔴  | ⬜     | PRD FR-12                  | argmax(magnitude) within 1 bin of f                                    |
| 6K.10 | T-SNE tab: output shape (N, 3)                 | 🔴  | ⬜     | PRD FR-10                  | TSNE result ndim==2 and shape[1]==3                                    |
| 6K.11 | PCA tab: explained_variance_ratio_ sums ≤ 1    | 🔴  | ⬜     | PRD FR-11                  | sum(explained_variance_ratio_[:3]) ≤ 1.0                               |
| 6K.12 | SWEEP NOISE interval callback executes         | 🟡  | ⬜     | PRD FR-9 §9.1              | Interval n_intervals increments → σ changes                            |
| 6K.13 | Freq slider respects Nyquist maximum (Fs/2)    | 🔴  | ⬜     | PRD FR-9 §9.2, PRD_sig §2  | Slider max updates dynamically when Fs changes                         |

---

## Phase 7 — Experiments + Lab-Report README

### 7A — Experiments

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 7A.1  | Generate canonical dataset (seed=42)          | 🔴  | ⬜     | PRD §15                    | data/dataset.npz produced                                       |
| 7A.2  | Train MLP to convergence                      | 🔴  | ⬜     | PRD goal 2                 | Best checkpoint saved; logs captured                            |
| 7A.3  | Train RNN to convergence                      | 🔴  | ⬜     | PRD goal 2                 | Best checkpoint saved; logs captured                            |
| 7A.4  | Train LSTM to convergence                     | 🔴  | ⬜     | PRD goal 2                 | Best checkpoint saved; logs captured                            |
| 7A.5  | Run noise-robustness sweep (5 σ levels)       | 🔴  | ⬜     | PRD FR-7                   | results/noise_robustness.png produced                           |
| 7A.6  | Run per-frequency MSE analysis                | 🔴  | ⬜     | PRD FR-7                   | results/per_frequency_mse.png produced                          |
| 7A.7  | Verify at least one model < 0.05 test MSE     | 🔴  | ⬜     | PRD goal 3                 | Comparison table shows passing entry                            |
| 7A.8  | Capture training console logs to file         | 🟡  | ⬜     | PLAN §8                    | results/freq_extractor.log present                              |

### 7B — README (Lab-Report) Sections

| #     | Section                                       | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 7B.1  | Project overview & objectives                 | 🔴  | ⬜     | PRD §1                     | 1–2 paragraphs + bullet objectives                              |
| 7B.2  | Theoretical background (Fourier, Nyquist)     | 🔴  | ⬜     | PRD_sig §2                 | Includes equations + Nyquist explanation                        |
| 7B.3  | RNN/LSTM theory + gating diagrams             | 🔴  | ⬜     | PRD_mod §5                 | Equations and conceptual figures                                |
| 7B.4  | Dataset-generation methodology                | 🔴  | ⬜     | PRD_sig §3,4               | Frequency-choice rationale + parameters                         |
| 7B.5  | Architecture descriptions w/ design choices   | 🔴  | ⬜     | PRD_mod §3-5               | Each model justified with trade-offs                            |
| 7B.6  | Training procedure & hyperparameters          | 🔴  | ⬜     | PRD_tr §5                  | All hyperparams listed with rationale                           |
| 7B.7  | Quantitative results (MSE table)              | 🔴  | ⬜     | PRD FR-7                   | MLP vs RNN vs LSTM train/val/test                               |
| 7B.8  | Training curves figure                        | 🔴  | ⬜     | PRD_tr §11                 | Embedded results/training_curves.png                            |
| 7B.9  | Prediction-comparison figures                 | 🔴  | ⬜     | PRD_tr §11                 | One per model; embedded                                         |
| 7B.10 | Noise-robustness analysis                     | 🔴  | ⬜     | PRD FR-7                   | Plot + commentary                                               |
| 7B.11 | Per-frequency analysis                        | 🔴  | ⬜     | PRD FR-7                   | Plot + which frequencies hardest                                |
| 7B.12 | Conclusions & observations                    | 🔴  | ⬜     | PRD §15                    | Which model wins + theoretical explanation                      |
| 7B.13 | Reproduction instructions                     | 🔴  | ⬜     | NFR-7                      | Exact commands + expected wall-clock time                       |
| 7B.14 | Repository link                               | 🔴  | ⬜     | PRD §15                    | GitHub URL prominently shown                                    |
| 7B.15 | Installation + uv quickstart                  | 🔴  | ⬜     | PRD §11                    | uv sync; uv run python src/main.py --mode all                   |
| 7B.16 | Configuration guide                           | 🔴  | ⬜     | NFR-7                      | Documented setup.json keys                                      |
| 7B.17 | Troubleshooting                               | 🟡  | ⬜     | NFR-7                      | At least 5 common failure modes                                 |
| 7B.18 | Testing instructions                          | 🔴  | ⬜     | NFR-5                      | uv run pytest tests/ --cov=src                                  |
| 7B.19 | Linting instructions                          | 🔴  | ⬜     | KPI                        | uv run ruff check .                                             |
| 7B.20 | Contribution guidelines                       | 🟢  | ⬜     | PRD §15                    | Branching, PR process                                           |
| 7B.21 | Credits + License (MIT)                       | 🔴  | ⬜     | PRD §15                    | MIT text or reference                                           |

---

## Phase 8 — Final Validation & Audit

| #     | Task                                          | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| 8.1   | uv lock                                       | 🔴  | ⬜     | UV-rules                   | uv.lock committed                                               |
| 8.2   | uv sync clean                                 | 🔴  | ⬜     | UV-rules                   | Fresh sync exits 0                                              |
| 8.3   | uv run ruff check . — 0 violations            | 🔴  | ⬜     | KPI                        | Ruff exit 0                                                     |
| 8.4   | uv run pytest tests/ — all pass               | 🔴  | ⬜     | NFR-5                      | pytest exit 0                                                   |
| 8.5   | uv run pytest tests/ --cov=src ≥ 85 %         | 🔴  | ⬜     | NFR-5                      | Coverage report ≥ 85                                            |
| 8.6   | grep secrets — 0 hits                         | 🔴  | ⬜     | NFR-4                      | No keys/tokens                                                  |
| 8.7   | Verify NO Python file > **145** code lines    | 🔴  | ⬜     | NFR-3 (v1.10)              | Counter script: 0 violations of new 145-line limit              |
| 8.8   | Verify all external I/O via Gatekeeper        | 🔴  | ⬜     | ADR-4                      | grep open() outside gatekeeper.py: 0 hits                       |
| 8.9   | Verify all business logic via SDK             | 🔴  | ⬜     | PLAN §1                    | main.py only calls sdk.* methods                                |
| 8.10  | Verify .env not committed                     | 🔴  | ⬜     | NFR-4                      | .gitignore covers .env                                          |
| 8.11  | Verify all PRDs / PLAN / TODO present         | 🔴  | ⬜     | PRD §15                    | All 7 docs/* files exist                                        |
| 8.12  | Verify results/*.png present (≥150 dpi)       | 🔴  | ⬜     | PRD §15                    | 7 PNGs                                                          |
| 8.13  | Produce final 32-row audit table              | 🔴  | ⬜     | All                        | All categories PASS                                             |
| 8.14  | Verify `--mode ui` launches without error     | 🔴  | ⬜     | PRD FR-13                  | Subprocess test: exit code 0 after graceful interrupt           |
| 8.15  | Verify ui_service*.py files ≤ 145 code lines  | 🔴  | ⬜     | NFR-3 (v1.10)              | All UI module files comply                                      |
| 8.16  | Verify 4 Dash tabs present in app layout      | 🔴  | ⬜     | PRD FR-9..12               | SIGNALS, T-SNE 3D, PCA 3D, FFT SPECTRUM all in layout          |
| 8.17  | Verify per-sinusoid controls (4 blocks)       | 🔴  | ⬜     | PRD FR-9 §9.2              | Sin 1..Sin 4 blocks rendered; MIX/BPF/f/φ/A controls present   |

---

## Edge Cases & Defensive Programming

| #     | Edge Case                                     | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| EC.1  | Empty dataset list                            | 🔴  | ⬜     | PRD_sig §11                | Raises ValueError with clear message                            |
| EC.2  | Single-frequency dataset                      | 🔴  | ⬜     | PRD_sig §11                | Splits succeed; one-hot still 4-dim                             |
| EC.3  | Extreme noise σ=10.0                          | 🟡  | ⬜     | PRD_tr §10                 | Training does not diverge (NaN-free)                            |
| EC.4  | window_size > total_samples                   | 🔴  | ⬜     | PRD_sig §11                | ValueError                                                      |
| EC.5  | sampling_rate violates Nyquist                | 🔴  | ⬜     | PRD_sig §2                 | ValueError at startup                                           |
| EC.6  | Negative frequency                            | 🔴  | ⬜     | PRD_sig §11                | ValueError                                                      |
| EC.7  | Zero amplitude (signal = 0)                   | 🟡  | ⬜     | PRD_sig §11                | Allowed; produces zero target                                   |
| EC.8  | Mismatched config_version on checkpoint load  | 🔴  | ⬜     | PLAN §5                    | ValueError; refuse to load                                      |
| EC.9  | Disk full / permission denied during save     | 🔴  | ⬜     | ADR-4                      | Gatekeeper retries then raises GatekeeperError                  |
| EC.10 | CUDA OOM on training start                    | 🟡  | ⬜     | PLAN §8                    | Falls back to CPU + logs warning                                |
| EC.11 | Very small dataset (< batch_size)             | 🟡  | ⬜     | PRD FR-6                   | drop_last=False; partial batch handled                          |
| EC.12 | All-zero noise vector                         | 🟢  | ⬜     | PRD_sig §11                | Tolerated (degenerate Gaussian)                                 |

---

## Data Validation Steps

| #     | Validation                                    | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| DV.1  | Schema check on setup.json keys               | 🔴  | ⬜     | PLAN §6                    | All required keys present; types match                          |
| DV.2  | Schema check on rate_limits.json              | 🔴  | ⬜     | PLAN §6                    | Required keys + numeric ranges validated                        |
| DV.3  | Schema check on logging_config.json           | 🔴  | ⬜     | PLAN §6                    | Required keys present; level ∈ {DEBUG..ERROR}                   |
| DV.4  | Hyperparameter ranges validated               | 🔴  | ⬜     | PRD_tr §5                  | lr > 0; batch_size > 0; epochs > 0                              |
| DV.5  | Frequency list non-empty + positive           | 🔴  | ⬜     | PRD_sig §3                 | len ≥ 1; all > 0                                                |
| DV.6  | sampling_rate ≥ 2·max(frequencies)            | 🔴  | ⬜     | PRD_sig §2                 | Nyquist enforced at startup                                     |
| DV.7  | Split ratios sum to 1.0 ± 1e-6                | 🔴  | ⬜     | PRD_sig §7                 | ValueError otherwise                                            |
| DV.8  | window_size ≤ total_samples                   | 🔴  | ⬜     | PRD_sig §11                | ValueError otherwise                                            |
| DV.9  | Tensor dtype = float32 throughout             | 🔴  | ⬜     | PLAN §5                    | Asserted in DataService                                         |
| DV.10 | No NaN/Inf in generated dataset               | 🔴  | ⬜     | PRD_sig §11                | torch.isfinite(x).all()                                         |
| DV.11 | Train/val/test disjoint by index              | 🔴  | ⬜     | PRD_sig §7                 | Index sets have empty intersection                              |
| DV.12 | One-hot label sums == 1                       | 🔴  | ⬜     | PRD FR-2                   | Asserted on every batch                                         |
| DV.13 | Normaliser stats finite + non-zero std        | 🔴  | ⬜     | PRD_sig §7                 | std > 1e-8                                                      |
| DV.14 | Saved dataset round-trip equality             | 🟡  | ⬜     | PLAN §5                    | np.load(npz) equals original arrays                             |

---

## Documentation Requirements (Final Lab Report)

| #     | Requirement                                   | Pri | Status | Refs                       | Definition of Done                                              |
|-------|-----------------------------------------------|-----|--------|----------------------------|-----------------------------------------------------------------|
| DOC.1 | All public functions have NumPy docstrings    | 🔴  | ⬜     | PRD_tr §13                 | Args/Returns/Raises documented                                  |
| DOC.2 | All public classes have docstrings            | 🔴  | ⬜     | PRD §11                    | Purpose + attributes documented                                 |
| DOC.3 | Type hints on every public signature          | 🔴  | ⬜     | NFR-2                      | mypy strict (or ruff TYPE) clean                                |
| DOC.4 | README contains all 21 sections (7B.1–7B.21)  | 🔴  | ⬜     | Phase 7B                   | Cross-checked vs section list                                   |
| DOC.5 | All figures captioned + referenced in README  | 🔴  | ⬜     | PRD_tr §11                 | Each PNG appears in body with caption                           |
| DOC.6 | Hyperparameter table with rationale           | 🔴  | ⬜     | PRD_tr §5                  | Every choice justified                                          |
| DOC.7 | Activation-function rationale per model       | 🔴  | ⬜     | PRD_mod §3-5               | ReLU/Tanh/Sigmoid choices explained                             |
| DOC.8 | Learning-rate + schedule rationale            | 🔴  | ⬜     | PRD_tr §5                  | Why 1e-3 + ReduceLROnPlateau                                    |
| DOC.9 | Batch-size rationale                          | 🔴  | ⬜     | PRD_tr §5                  | Why 32 (memory + variance trade-off)                            |
| DOC.10| Best-model justification w/ theory            | 🔴  | ⬜     | PRD goal 4                 | Theoretical + empirical evidence                                |
| DOC.11| Limitations + future work                     | 🟡  | ⬜     | PRD §15                    | Honest assessment included                                      |
| DOC.12| ADR list inlined or linked                    | 🟡  | ⬜     | PLAN §9                    | All architectural decisions recorded                            |
| DOC.13| docs/PROMPTS.md kept current                  | 🟡  | ⬜     | PRD §15                    | Records all prompts used                                        |

---

## Known Risks

| Risk                                       | Likelihood | Impact | Mitigation                                                        |
|--------------------------------------------|------------|--------|-------------------------------------------------------------------|
| LSTM overfits small dataset                | Medium     | Medium | Dropout 0.1, early stopping (patience=10)                         |
| RNN diverges (exploding gradient)          | Low        | High   | Gradient clipping max_norm=1.0; orthogonal init                   |
| Coverage drops below 85 %                  | Low        | High   | Write tests first (TDD); per-module ≥ 90 % targets                |
| Windows path issues                        | Medium     | Low    | Use pathlib.Path throughout                                       |
| File creeps over 145-line limit            | Medium     | Medium | Pre-commit hook + Phase 8.7 audit                                 |
| Random seed not honoured by CUDA           | Low        | Medium | Document non-determinism in README; force CPU for tests           |
| Config schema drift between versions       | Low        | High   | Version field + parse_version compatibility check                 |
| Gatekeeper deadlock on retries             | Low        | High   | Bounded queue + max_retries cap                                   |
| Numerical instability with extreme noise   | Medium     | Medium | Input clipping; log-domain when feasible                          |
| matplotlib backend missing on headless env | Medium     | Low    | Set Agg backend in evaluation_service                             |

---

## Definition of Done (Global)

A task is DONE only when **all** the following hold:

1. **Code** is written and merged on `main`.
2. **Ruff** reports 0 violations against the file.
3. **Pytest** passes locally with the relevant tests added/updated.
4. **Coverage** is verified ≥ 85 % globally and ≥ 90 % for the changed module.
5. **Docstrings** (NumPy style) are present on every public function/class.
6. **Type hints** annotate every public signature.
7. **No secrets** (keys, tokens, passwords, paths from `.env`) are hard-coded.
8. **No business logic** is duplicated outside the SDK.
9. **All external I/O** is funnelled through `Gatekeeper`.
10. **File length** ≤ **145 code lines** (excluding blanks/comments) — strict per NFR-3 v1.10.
11. **Cross-references** to PRD/PLAN are added in this TODO row.
12. **Logged** in `docs/PROMPTS.md` if produced via AI assistance.

---

## Changelog

| Version | Date       | Author | Change                                                                       |
|---------|------------|--------|------------------------------------------------------------------------------|
| 1.00    | 2026-04-26 | Dev    | Initial TODO derived from PRD/PLAN.                                          |
| 1.10    | 2026-04-28 | Dev    | Decomposed all phases into atomic sub-tasks; added Edge Cases, Data         |
|         |            |        | Validation, Documentation, expanded Risks; reduced max file length to 145.  |
| 1.20    | 2026-04-28 | Dev    | Added Phase 4F (UIService / Sinusoid Explorer) with 4F-1..4F-6 sub-tasks;   |
|         |            |        | added Phase 6K (13 UIService unit tests); added Phase 8.14..8.17 UI audit   |
|         |            |        | items; aligned with PRD v1.20 and PLAN v1.10 (FR-9..FR-13, ADR-6).          |
