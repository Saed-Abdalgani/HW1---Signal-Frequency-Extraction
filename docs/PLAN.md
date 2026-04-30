# Architecture and Implementation Plan
## HW1 — Signal Frequency Extraction using Neural Networks

| Field    | Value               |
|----------|---------------------|
| Version  | 1.10                |
| Updated  | 2026-04-28          |

---

## 1. Architecture Overview

The system follows a strict layered architecture with a single SDK entry point:

```
CLI (src/main.py)
       │
       ▼
SDK Layer (src/freq_extractor/sdk/sdk.py)        ← single entry point for all logic
       │
       ├─► DataService       (signal generation, dataset construction)
       ├─► ModelFactory      (instantiate MLP / RNN / LSTM)
       ├─► TrainingService   (training loop, early stopping, checkpointing)
       ├─► EvaluationService (metrics, plots, comparison table)
       └─► UIService         (Sinusoid Explorer Dash app — FR-9..FR-13)
               │
               ▼
       Shared Infrastructure
       ├─► Config            (load + validate config/*.json)
       ├─► Gatekeeper        (rate limiting, retry, logging for I/O ops)
       └─► Version           (version validation)
```

**Dependency Direction:**
```
CLI → SDK → Services → Shared/Infrastructure
```
No service imports from the CLI layer.  No service imports `sdk.py` (avoids circular imports).
`UIService` calls `DataService` directly (signal generation) and is orchestrated only via SDK.

---

## 2. C4 Context Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    freq_extractor System                  │
│                                                           │
│  ┌──────────┐    CLI args     ┌─────────────────────┐   │
│  │  User /  │ ──────────────► │   src/main.py        │   │
│  │  Grader  │                 │   (Entry Point)      │   │
│  └──────────┘                 └─────────┬───────────┘   │
│                                         │ calls SDK      │
│                                         ▼               │
│                               ┌─────────────────────┐   │
│                               │    sdk/sdk.py        │   │
│                               │  (Business Logic Hub)│   │
│                               └─────────────────────┘   │
│                                                           │
│  External Systems: NONE (fully self-contained)            │
└─────────────────────────────────────────────────────────┘
```

---

## 3. C4 Container Diagram

```
┌─────────────────────────────── freq_extractor Package ────────┐
│                                                                 │
│  ┌─────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │  DataService │  │ ModelFactory   │  │ TrainingService   │   │
│  │             │  │                │  │                  │   │
│  │ - generate  │  │ - create_model │  │ - train()        │   │
│  │ - build_ds  │  │ - MLPModel     │  │ - early_stop     │   │
│  │ - split     │  │ - RNNModel     │  │ - checkpoint     │   │
│  │ - normalize │  │ - LSTMModel    │  │ - lr_schedule    │   │
│  └──────┬──────┘  └───────┬────────┘  └───────┬──────────┘   │
│         │                 │                    │               │
│         └─────────────────┴────────────────────┘              │
│                           │                                     │
│                           ▼                                     │
│    ┌──────────────────┐  ┌───────────────────────────────┐    │
│    │  EvaluationSvc   │  │  UIService (Sinusoid Explorer) │    │
│    │ - compute_mse    │  │ - build_app()  (Dash layout)   │    │
│    │ - plot_curves    │  │ - _signals_tab()               │    │
│    │ - plot_preds     │  │ - _tsne_tab()                  │    │
│    │ - compare_table  │  │ - _pca_tab()                   │    │
│    └──────────────────┘  │ - _fft_tab()                   │    │
│                           │ - _register_callbacks()        │    │
│                           └───────────────────────────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                  Shared Infrastructure                    │ │
│  │  Config │ Gatekeeper │ Version │ Constants               │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. C4 Component Diagram: DataService

```
DataService
├── SignalGenerator         — produces time array + clean + noisy signal
├── DatasetBuilder          — sliding window, one-hot encoding
├── DatasetSplitter         — stratified train/val/test split
├── DataNormalizer          — fit on train, transform all splits
└── DataPersistence         — save/load .npz files via Gatekeeper
```

---

## 5. Data Schema

### Dataset Entry
```python
{
    "frequency_label": np.ndarray,  # shape (4,), one-hot float32
    "noisy_samples":   np.ndarray,  # shape (10,), float32, normalized
    "clean_samples":   np.ndarray,  # shape (10,), float32, normalized
    "target_output":   np.ndarray,  # shape (1,),  float32, normalized
}
```

### Model Input Tensors
```python
# MLP
X_mlp:  torch.Tensor  # shape (B, 14)  — [noisy_samples ‖ freq_label]
y:      torch.Tensor  # shape (B, 1)   — target_output

# RNN / LSTM
X_seq:  torch.Tensor  # shape (B, 10, 5) — each step: [sample ‖ freq_label]
y:      torch.Tensor  # shape (B, 1)
```

### Checkpoint Schema
```python
{
    "model_state_dict":     dict,
    "optimizer_state_dict": dict,
    "epoch":                int,
    "val_mse":              float,
    "config_version":       str,
    "model_type":           str,   # "mlp" | "rnn" | "lstm"
}
```

---

## 6. Module Responsibilities

| Module                           | Responsibility                                                         |
|----------------------------------|------------------------------------------------------------------------|
| `sdk/sdk.py`                     | Orchestrate all workflows; single public API                           |
| `services/data_service.py`       | Signal generation, dataset build, split, normalize                     |
| `services/mlp_model.py`          | MLPModel class definition                                              |
| `services/rnn_model.py`          | RNNModel class definition                                              |
| `services/lstm_model.py`         | LSTMModel class definition                                             |
| `services/model_factory.py`      | Create model by name; manage model registry                            |
| `services/training_service.py`   | Training loop, early stopping, checkpointing                           |
| `services/evaluation_service.py` | Metrics, static PNG plots, result tables                               |
| `services/ui_service.py`         | Sinusoid Explorer Dash app; layout + callbacks (FR-9..FR-13)           |
| `shared/config.py`               | Load, validate, cache config/*.json                                    |
| `shared/gatekeeper.py`           | Centralized I/O gatekeeper (rate limit, retry, log)                    |
| `shared/version.py`              | Version constants + config compatibility check                         |
| `constants.py`                   | Global constants (MODEL_TYPES, SPLIT_NAMES, DISPLAY_MODES, etc.)       |
| `main.py`                        | CLI argument parsing → SDK calls only (including `--mode ui`)          |

---

## 7. Architectural Decision Records (ADRs)

### ADR-1: Pure PyTorch (no Lightning / Keras)
**Decision:** Use raw PyTorch training loops.  
**Rationale:** Demonstrates understanding of training mechanics; no hidden abstractions; fits educational context.  
**Trade-off:** More boilerplate code; mitigated by TrainingService abstraction.

### ADR-2: Tanh for MLP, default for RNN/LSTM
**Decision:** MLP uses Tanh activations.  RNN/LSTM use their built-in tanh/sigmoid gates.  
**Rationale:** Sinusoidal outputs are bounded ≈ [-1, 1]; Tanh naturally aligns with this range.  ReLU risks unbounded outputs.

### ADR-3: Adam optimizer for all models
**Decision:** Adam with lr=0.001 for all architectures.  
**Rationale:** Reduces confounding variables when comparing architectures; Adam converges reliably.

### ADR-4: Gatekeeper for file I/O
**Decision:** All file read/write operations pass through `ApiGatekeeper`.  
**Rationale:** Architecture rule compliance; allows future extension to remote storage; provides unified logging.

### ADR-5: Config-driven hyperparameters
**Decision:** All hyperparameters, frequencies, paths in `config/setup.json`.
**Rationale:** Reproducibility, configurability, no hardcoded magic numbers.

### ADR-6: Plotly Dash for Interactive UI
**Decision:** Use Plotly Dash (Python-native) for the Sinusoid Explorer dashboard.
**Rationale:** Pure-Python stack compatible with `uv`; Plotly natively supports 3D scatter (T-SNE, PCA) and FFT magnitude plots; Dash's callback model cleanly separates UI state from signal logic; no JavaScript required.
**Trade-off:** Dash dev server only (no production WSGI) — acceptable for educational context per FR-13.

---

## 8. Error Handling Design

| Layer          | Error Handling Strategy                                          |
|----------------|------------------------------------------------------------------|
| Config loading | Raise `ConfigurationError` on missing keys or version mismatch  |
| Data generation| Validate all parameters, raise `ValueError` on invalid input    |
| Training       | Catch CUDA OOM, fall back to CPU; log all exceptions            |
| File I/O       | Gatekeeper retries on transient failures; raises after max_retries |
| CLI            | Argparse validation; catch all exceptions, print user-friendly message |

---

## 9. Security Architecture

- No external APIs or network calls.
- `os.environ.get()` used for any sensitive future configuration.
- `.env` never committed; `.env-example` provided with placeholders.
- File paths constructed with `pathlib.Path`, never with `os.path.join` on user input.

---

## 10. Testing Strategy

| Layer        | Type        | Tool          | Coverage Target |
|--------------|-------------|---------------|-----------------|
| Services     | Unit        | pytest + mock | ≥ 90 %          |
| Shared       | Unit        | pytest        | ≥ 90 %          |
| SDK          | Integration | pytest        | ≥ 85 %          |
| Training     | Integration | pytest + tmp  | ≥ 85 %          |

All tests are deterministic (fixed seed).  External I/O is mocked.  No test touches real network or GPU (CI-safe).

---

## 11. Extension Points

| Extension              | How to Add                                                            |
|------------------------|-----------------------------------------------------------------------|
| New model architecture | Add `services/<name>_model.py`, register in `ModelFactory`            |
| New frequency          | Update `config/setup.json` → `frequencies` array                      |
| New metric             | Add method to `EvaluationService`                                     |
| Remote data storage    | Extend `Gatekeeper` with HTTP adapter                                 |
| New CLI command        | Add argparse subcommand; add SDK method                               |
| New UI tab             | Add `_<name>_tab()` to `UIService`; register callback; add to layout  |
| New noise model        | Add option to `DataService.generate()`; expose in UI Noise dropdown   |
| New filter type        | Implement filter in `DataService`; add option to UI Filter dropdown   |
