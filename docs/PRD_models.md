# PRD: Neural Network Model Architectures

| Field    | Value               |
|----------|---------------------|
| Version  | 1.00                |
| Parent   | docs/PRD.md         |
| Updated  | 2026-04-28          |

---

## 1. Description

Three neural network architectures are implemented to solve the frequency extraction task.  Each architecture reflects different assumptions about how temporal context should be encoded.  This document specifies the input/output contract, internal design, and expected behavior for each model.

---

## 2. Shared Input/Output Contract

| Aspect          | MLP                              | RNN / LSTM                             |
|-----------------|----------------------------------|----------------------------------------|
| Batch shape     | (B, 14)                          | (B, 10, 5)                             |
| Input meaning   | [x₀..x₉ ‖ one-hot]              | timestep t: [xₜ ‖ one-hot]            |
| Output shape    | (B, 1)                           | (B, 1)                                 |
| Output meaning  | Predicted next clean sample      | Predicted next clean sample            |

Notation: B = batch size, 14 = 10 samples + 4 one-hot, 5 = 1 sample + 4 one-hot, 10 = window size.

---

## 3. Model 1: Fully Connected MLP

### Purpose
Provides a non-sequential baseline.  Treats the 10-sample window as a flat feature vector with no explicit temporal ordering.

### Architecture
```
Input:  (B, 14)      — 10 noisy samples + 4-dim one-hot
Linear: 14 → 64      — first hidden layer
Tanh activation
Linear: 64 → 128     — second hidden layer (wider for richer features)
Tanh activation
Linear: 128 → 64     — bottleneck
Tanh activation
Linear: 64 → 1       — scalar output
Output: (B, 1)
```

### Design Rationale
- Tanh preferred over ReLU for bounded sinusoidal targets (output ∈ [-1, 1] approx.).
- Width 128 in middle layer allows the network to learn Fourier-like basis functions.
- 4-layer depth balances expressivity and overfitting risk for this small problem.

### Expected Behavior
- Will learn frequency-specific offsets from the one-hot label.
- Limited by absence of temporal ordering — treats samples as a bag of features.
- Should achieve moderate MSE but be outperformed by sequential models for most frequencies.

---

## 4. Model 2: RNN (Vanilla Recurrent Neural Network)

### Purpose
Tests whether short-term sequential memory is sufficient for frequency extraction.  The hidden state carries temporal context from step t-W to t.

### Architecture
```
Input per step: (B, 5)   — [noisy_sample ‖ one-hot]
RNN layer:  input_size=5, hidden_size=64, num_layers=2, batch_first=True
           → h_t at final timestep: (B, 64)
Linear:    64 → 1
Output:    (B, 1)
```

### Design Rationale
- 2 RNN layers allow hierarchical temporal abstraction.
- Hidden size 64 matches MLP to keep parameter count comparable.
- `batch_first=True` for consistency with PyTorch convention.
- Dropout = 0.1 between layers to reduce overfitting.

### Expected Behavior
- Strong for high frequencies (f₃=30 Hz, f₄=50 Hz) where multiple cycles fit in 10-sample window.
- Weaker for f₁=5 Hz (only 0.25 cycles in 10 samples at 200 Hz) — gradient vanishing prevents long-range recall.
- Should outperform MLP because temporal ordering is preserved.

### Vanishing Gradient Note
Standard RNN suffers gradient vanishing for sequences > ~10 steps.  Since our window is exactly 10 steps, performance degradation is expected at the boundary.  This makes LSTM comparison meaningful.

---

## 5. Model 3: LSTM (Long Short-Term Memory)

### Purpose
Tests whether gated memory cells improve frequency extraction, especially for low-frequency signals requiring context across the full 10-step window.

### Architecture
```
Input per step: (B, 5)   — [noisy_sample ‖ one-hot]
LSTM layer: input_size=5, hidden_size=64, num_layers=2, batch_first=True
           → h_t at final timestep: (B, 64)
Linear:    64 → 1
Output:    (B, 1)
```

### LSTM Gates (per cell, per layer)
```
Input gate:  iₜ = σ(Wᵢxₜ + Uᵢhₜ₋₁ + bᵢ)
Forget gate: fₜ = σ(Wᶠxₜ + Uᶠhₜ₋₁ + bᶠ)
Cell update: g̃ₜ = tanh(Wᵍxₜ + Uᵍhₜ₋₁ + bᵍ)
Cell state:  Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ g̃ₜ
Output gate: oₜ = σ(Wᵒxₜ + Uᵒhₜ₋₁ + bᵒ)
Hidden:      hₜ = oₜ ⊙ tanh(Cₜ)
```

### Design Rationale
- Forget gate allows the network to selectively discard irrelevant past information — critical for multi-scale frequency patterns.
- Cell state Cₜ acts as a conveyor belt for long-range gradients, solving RNN's vanishing-gradient problem.
- Same hidden size (64) and layers (2) as RNN for fair comparison.
- Dropout = 0.1 between LSTM layers.

### Expected Behavior
- Best overall performance due to gated memory.
- Handles both low-frequency (f₁=5 Hz) and high-frequency (f₄=50 Hz) well.
- Training is slower per epoch than MLP due to 4× gate computations.
- Should achieve lowest test MSE overall.

---

## 6. Parameter Count Comparison

| Model | Parameters (approx.)        |
|-------|-----------------------------|
| MLP   | 14×64 + 64×128 + 128×64 + 64×1 ≈ 17,601 |
| RNN   | 2-layer RNN with h=64: ≈ 25,473          |
| LSTM  | 2-layer LSTM with h=64: ≈ 99,585         |

LSTM has 4× more parameters than RNN because of 4 gate matrices per cell.

---

## 7. Weight Initialization

All models use PyTorch defaults:
- Linear layers: Kaiming uniform (He initialization).
- RNN/LSTM: Orthogonal initialization for recurrent weights (reduces gradient issues).
- Biases: Zero-initialized.

---

## 8. Alternatives Considered

| Alternative                    | Rejected Because                                            |
|--------------------------------|-------------------------------------------------------------|
| GRU (Gated Recurrent Unit)     | Less illustrative of gate mechanisms; LSTM is canonical     |
| Transformer                    | Out of scope; overkill for 10-step window                   |
| CNN (1D Convolution)           | Not specified; frequency convolution is implicit            |
| Deeper MLP (6+ layers)         | Risk of overfitting; specification says document rationale  |

---

## 9. Test Scenarios

| Test ID | Scenario                                    | Success Criterion                        |
|---------|---------------------------------------------|------------------------------------------|
| MD-T1   | MLP forward pass, batch=32                  | Output shape (32, 1), no NaN             |
| MD-T2   | RNN forward pass, batch=32, seq=10          | Output shape (32, 1), no NaN             |
| MD-T3   | LSTM forward pass, batch=32, seq=10         | Output shape (32, 1), no NaN             |
| MD-T4   | All models accept batch size = 1            | No crash, valid output                   |
| MD-T5   | Gradient flows through all layers           | No zero-grad after backward pass         |
| MD-T6   | Model factory creates correct type          | `create_model("mlp")` returns MLPModel   |
| MD-T7   | LSTM has more params than RNN               | param_count(LSTM) > param_count(RNN)     |

---

## 10. Success Criteria

1. All three models produce outputs of shape (B, 1) for valid inputs.
2. Gradients flow through all parameters without NaN or zero.
3. Model parameter counts match theoretical calculations (within 5%).
4. Each model is serializable with `torch.save` and loadable with `torch.load`.
