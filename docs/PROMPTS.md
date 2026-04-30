# AI-Assisted Development Log (PROMPTS.md)
## HW1 — Signal Frequency Extraction using Neural Networks

| Field    | Value               |
|----------|---------------------|
| Version  | 1.00                |
| Updated  | 2026-04-28          |

---

## Purpose

This document records all significant AI-assisted development interactions, their context, purpose, output summaries, and lessons learned.  It provides transparency about AI contributions and supports reproducibility of the development process.

---

## Prompt 1 — Full Project Bootstrap

**Tool:** Augment Agent (Claude Sonnet 4.6)  
**Date:** 2026-04-28  
**Context:** Empty workspace, complete project specification provided.

**Prompt Summary:**
```
You are a Senior Software Architect... [full system prompt with 22 rules]
PROJECT DETAILS: HW1 — Signal Frequency Extraction using Neural Networks
[Full project specification with theoretical background, dataset spec,
network architectures, requirements checklist, and success criteria]
```

**Purpose:** Bootstrap the entire project from scratch — documentation, scaffolding, implementation, tests, and final validation.

**Output Summary:**
- Created docs/PRD.md (full product requirements, 15 sections)
- Created docs/PLAN.md (C4 diagrams, ADRs, data schemas, module map)
- Created docs/TODO.md (44 tasks across 8 phases with priorities)
- Created docs/PRD_signal_generation.md (signal generation subsystem PRD)
- Created docs/PRD_models.md (MLP, RNN, LSTM architecture specifications)
- Created docs/PRD_training.md (loss, optimizer, early stopping, checkpointing)
- Created docs/PROMPTS.md (this file)
- Created all config/*.json, pyproject.toml, .env-example, .gitignore
- Implemented full Python package: data_service, mlp/rnn/lstm models, training_service, evaluation_service, sdk, main.py
- Created comprehensive test suite: unit + integration tests
- Ran experiments: trained all 3 models, generated all result plots
- Final README as complete lab report

**Iterations:** 1 major prompt with iterative file-by-file refinement

**Key Design Decisions Made by AI:**
1. Frequencies: 5, 15, 30, 50 Hz — covers 3 octaves, max 50 Hz → 200 Hz sampling rate.
2. Tanh activations for MLP (bounded sinusoidal outputs).
3. Adam optimizer for all models (reduces confounding variables).
4. Gradient clipping max_norm=1.0 (prevents RNN exploding gradients).
5. Gatekeeper wraps file I/O (architecture compliance + future extensibility).
6. ReduceLROnPlateau scheduler (adaptive without fixed schedule).

**Lessons Learned:**
- Documentation-first discipline forces careful architecture thinking before coding.
- Fixed seeds must be set in 4 places (torch, numpy, random, cudnn) for full reproducibility.
- File-size limits (150 lines) require splitting models into separate files.
- Tanh > ReLU for bounded regression targets (sinusoidal signals).

---

## Prompt 2 — Architecture Clarification

**Tool:** Augment Agent (in-context refinement)  
**Date:** 2026-04-28  
**Context:** Deciding how to handle Gatekeeper for a project with no external APIs.

**Question Asked:**
"The architecture rules require a Gatekeeper for all external API calls, but this project has no external APIs. How should the Gatekeeper be implemented?"

**Resolution:**
The Gatekeeper was implemented to wrap all file I/O operations (dataset save/load, checkpoint save/load, results save).  This satisfies the architecture rule while being practically meaningful — file operations can fail transiently (disk full, locked files) and benefit from retry logic and logging.  The Gatekeeper is also designed as an extension point for future remote storage adapters.

**Output:** `src/freq_extractor/shared/gatekeeper.py` with execute(), retry logic, and get_queue_status().

---

## Prompt 3 — Test Design

**Tool:** Augment Agent (in-context)  
**Date:** 2026-04-28  
**Context:** Writing unit tests for signal generation with FFT verification.

**Key Insight:**
Tests for signal correctness use `scipy.fft` to verify that the dominant frequency in the generated signal matches the configured frequency.  This is a white-box test that validates the mathematical correctness of the generated data, not just its shape.

**Test Pattern Used:**
```python
freqs = scipy.fft.rfftfreq(n_samples, d=1/sampling_rate)
magnitudes = np.abs(scipy.fft.rfft(signal))
dominant_freq = freqs[np.argmax(magnitudes)]
assert abs(dominant_freq - expected_freq) < 0.5  # within 0.5 Hz
```

---

## Prompt 4 — Hyperparameter Justification

**Tool:** Augment Agent (in-context)  
**Date:** 2026-04-28  
**Context:** Documenting rationale for all hyperparameter choices.

**Decisions Documented:**

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Hidden size    | 64    | Balances expressivity and training speed; 3× input features |
| Num layers     | 2     | Hierarchical abstraction; 3+ layers risk vanishing gradients in RNN |
| Batch size     | 64    | Stable gradient estimates; fits in RAM; standard for small datasets |
| Learning rate  | 0.001 | Adam paper default; empirically validated for sequence models |
| Max epochs     | 300   | Generous upper bound; early stopping prevents waste |
| ES patience    | 20    | 20 × batch_iterations ≈ enough to escape local plateaus |
| Dropout        | 0.1   | Light regularization; dataset is small (7960 entries) |
| Grad clip      | 1.0   | Standard for RNNs; prevents gradient explosion in BPTT |

---

## Best Practices Identified

1. **Documentation before code**: PRD → PLAN → TODO → code dramatically reduces rework.
2. **Config-driven everything**: Changing an experiment requires only editing JSON, not source code.
3. **Fixed seeds everywhere**: torch + numpy + random + cudnn for true reproducibility.
4. **Separate model files**: Keeping MLP, RNN, LSTM in separate files respects 150-line limit.
5. **TDD mindset**: Writing test scenarios in PRDs before implementation guides better API design.
6. **Gatekeeper pattern**: Even for file I/O, centralizing operations enables consistent logging.
7. **Tanh for regression on bounded signals**: ReLU can produce unbounded outputs; Tanh is safer.
8. **Gradient clipping**: Always clip gradients in RNN/LSTM training to prevent divergence.
