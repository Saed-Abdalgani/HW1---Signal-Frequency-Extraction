# PRD: Training Procedure and Hyperparameter Strategy

| Field    | Value               |
|----------|---------------------|
| Version  | 1.00                |
| Parent   | docs/PRD.md         |
| Updated  | 2026-04-28          |

---

## 1. Description

This document specifies the training procedure, loss function, optimizer configuration, hyperparameter choices, early stopping strategy, checkpointing, and evaluation protocol applied uniformly to all three neural network architectures (MLP, RNN, LSTM).

---

## 2. Loss Function

**Mean Squared Error (MSE)**
```
L = (1/B) Σᵢ (ŷᵢ − yᵢ)²
```
- `ŷᵢ` : model prediction of the next clean sample.
- `yᵢ` : ground truth next clean sample.
- MSE is appropriate because:
  - Target is a continuous scalar (regression task).
  - Squared penalty heavily penalizes large prediction errors.
  - Direct interpretability: units are (amplitude)².

PyTorch implementation: `torch.nn.MSELoss(reduction='mean')`.

---

## 3. Optimizer

**Adam (Adaptive Moment Estimation)**
```
θₜ = θₜ₋₁ − α · m̂ₜ / (√v̂ₜ + ε)
```
- Learning rate α = 0.001 (configurable).
- β₁ = 0.9, β₂ = 0.999, ε = 1e-8 (PyTorch defaults).

**Rationale for Adam over SGD:**
- Adaptive per-parameter learning rates accelerate convergence on sparse gradients.
- Less sensitive to learning rate choice than SGD + momentum.
- Standard baseline for sequence modeling tasks in deep learning literature.

**Alternative considered: SGD + momentum (0.9)**  
Rejected: requires more careful LR tuning and scheduler design; Adam converges faster with default settings for this problem size.

---

## 4. Learning Rate Schedule

**ReduceLROnPlateau**
- Monitor: validation MSE.
- Factor: 0.5 (halve LR when plateau detected).
- Patience: 10 epochs (no improvement before reducing).
- Minimum LR: 1e-6.

Rationale: Prevents oscillation around minima late in training without requiring a fixed schedule that may not generalize across all three architectures.

---

## 5. Hyperparameter Configuration

All hyperparameters are read from `config/setup.json` — none are hardcoded.

| Hyperparameter      | Default Value | Configurable | Rationale                                            |
|---------------------|---------------|--------------|------------------------------------------------------|
| batch_size          | 64            | Yes          | Large enough for stable gradients, fits in RAM       |
| max_epochs          | 300           | Yes          | Upper bound; early stopping triggers before this     |
| learning_rate       | 0.001         | Yes          | Adam default; shown to work well for RNN/LSTM        |
| early_stop_patience | 20            | Yes          | Allows sufficient plateau exploration                |
| lr_reduce_patience  | 10            | Yes          | Half of early stop patience                          |
| hidden_size         | 64            | Yes          | Balanced expressivity vs. parameter count            |
| num_layers          | 2             | Yes          | Hierarchical temporal abstraction                    |
| dropout             | 0.1           | Yes          | Light regularization; dataset is small               |
| seed                | 42            | Yes (CLI)    | Reproducibility                                      |

---

## 6. Early Stopping

**Algorithm:**
1. Track best validation MSE across epochs.
2. If val_MSE does not improve by `min_delta=1e-6` for `patience=20` consecutive epochs, stop.
3. Restore model weights from the best epoch before stopping.

**Rationale:**
- Prevents overfitting without manual epoch selection.
- Best-epoch restoration ensures evaluation is on the true optimum.

---

## 7. Training Loop (Pseudocode)

```
for epoch in 1..max_epochs:
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = MSELoss(pred, y_batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        optimizer.step()

    model.eval()
    val_loss = compute_mse(model, val_loader)
    scheduler.step(val_loss)
    early_stopper.step(val_loss, model)

    if early_stopper.should_stop:
        break
```

**Gradient Clipping** (`max_norm=1.0`):  
Required for RNN to prevent exploding gradients during backpropagation through time (BPTT).  Applied to all models for consistency.

---

## 8. Reproducibility Protocol

1. `torch.manual_seed(seed)` — PyTorch random state.
2. `numpy.random.seed(seed)` — NumPy random state.
3. `random.seed(seed)` — Python built-in random.
4. `torch.backends.cudnn.deterministic = True` — GPU determinism.
5. Fixed `worker_init_fn` in DataLoaders.

Seed is set once at program startup via `src/freq_extractor/shared/config.py`.

---

## 9. Checkpointing

- Best model (lowest val_MSE) saved to `results/checkpoints/<model_name>_best.pt`.
- Checkpoint contains: `{model_state_dict, optimizer_state_dict, epoch, val_mse, config}`.
- Loading from checkpoint is supported via `TrainingService.load_checkpoint()`.

---

## 10. Evaluation Protocol

After training, each model is evaluated on the held-out test set:

| Metric    | Computation                                        |
|-----------|----------------------------------------------------|
| Test MSE  | Average MSE over all test batches                  |
| Train MSE | Final epoch training MSE                           |
| Val MSE   | Best validation MSE (from best checkpoint)         |

**Noise Robustness Experiment:**
- Fix model, vary σ ∈ {0.05, 0.10, 0.20, 0.30, 0.50}.
- Re-generate test sets at each σ.
- Plot Test MSE vs. σ for all three models.

**Per-Frequency Analysis:**
- Compute test MSE separately for each frequency label.
- Reveals which frequencies are harder to extract.

---

## 11. Visualization Outputs

| Plot                      | Filename                              | Description                              |
|---------------------------|---------------------------------------|------------------------------------------|
| Training curves           | `results/training_curves.png`         | Val/Train MSE vs. epoch for all models   |
| Prediction example (MLP)  | `results/predictions_mlp.png`         | Noisy input, MLP pred, clean GT          |
| Prediction example (RNN)  | `results/predictions_rnn.png`         | Same for RNN                             |
| Prediction example (LSTM) | `results/predictions_lstm.png`        | Same for LSTM                            |
| Noise robustness          | `results/noise_robustness.png`        | Test MSE vs. σ for all models            |
| Per-frequency MSE         | `results/per_frequency_mse.png`       | Bar chart, all models × all frequencies  |
| Signal examples           | `results/signal_examples.png`         | Clean, noisy, and target for each freq   |

---

## 12. Test Scenarios

| Test ID | Scenario                                    | Success Criterion                           |
|---------|---------------------------------------------|---------------------------------------------|
| TR-T1   | Training loop runs for 2 epochs (smoke test)| No exception, loss decreases                |
| TR-T2   | Early stopping triggers at patience=2        | Training stops after 2 epochs of no improvement |
| TR-T3   | Gradient clipping prevents NaN loss          | Loss stays finite through all epochs        |
| TR-T4   | Checkpoint saves/loads correctly             | Loaded model produces identical predictions |
| TR-T5   | LR scheduler reduces LR on plateau          | LR halves after 10 no-improvement epochs    |
| TR-T6   | MSE is zero for perfect predictions          | MSELoss(y, y) == 0.0                        |
| TR-T7   | Reproducibility: two runs same seed          | Identical final val_mse                     |

---

## 13. Success Criteria

1. All three models converge (validation loss decreases monotonically for at least 50 epochs).
2. At least one model achieves test MSE < 0.05.
3. Training completes within 10 minutes per model on CPU.
4. Checkpoints are loadable and reproduce identical test MSE.
5. All visualizations are saved at ≥ 150 dpi.
