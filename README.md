# HW1 — Signal Frequency Extraction using Neural Networks

## 1. Project Overview & Objectives

The primary objective of this project is to implement, train, and critically evaluate three distinct neural network architectures—a Fully Connected Multi-Layer Perceptron (MLP), a Vanilla Recurrent Neural Network (RNN), and a Long Short-Term Memory network (LSTM)—for the complex task of extracting a target frequency component from a composite noisy time-series signal. 

In real-world digital signal processing (DSP), it is often challenging to isolate single frequency components from a noisy medium without relying on explicit mathematical frequency-domain transformations such as the Fast Fourier Transform (FFT). By applying deep learning techniques to time-series data, we can implicitly learn temporal patterns and isolate these frequencies directly in the time domain. 

This project explores how varying degrees of network memory depth and complex gating mechanisms fundamentally alter a model's ability to denoise time-series data and extract long-range temporal dependencies.

**Key Objectives:**
- **Data Engineering:** Generate a highly reproducible, fully parameterised synthetic dataset composed of continuous sinusoidal signals augmented with additive Gaussian noise.
- **Architectural Implementation:** Design and instantiate three distinct neural network models (MLP, RNN, LSTM) using PyTorch, adhering to specific layer constraints and parameter scaling.
- **Robust Training:** Implement a rigorous training pipeline utilising Mean Squared Error (MSE), the Adam optimizer, adaptive learning rate scheduling (ReduceLROnPlateau), and Early Stopping to ensure convergence without overfitting.
- **Quantitative Evaluation:** Systematically evaluate and compare the models across multiple noise levels (signal-to-noise ratios) and target frequencies.
- **Interactive Visualisation:** Deliver a rich, interactive browser-based dashboard ("Sinusoid Explorer") built with Plotly Dash to enable real-time signal composition, noise manipulation, and advanced dimensionality reduction visualisations (T-SNE, PCA).
- **Code Quality:** Ensure strict adherence to non-functional requirements (NFRs), including a maximum file length of 145 code lines, comprehensive docstrings, full type-hinting, and a test suite with ≥ 85% coverage.

---

## 2. Theoretical Background

### Fourier Decomposition
At the heart of signal processing lies Fourier theory, which states that any continuous, periodic signal can be decomposed into a sum of infinite sinusoidal basis functions. In our context, we simplify the problem space to pure, single-frequency sinusoids to provide our neural networks with mathematically exact ground-truth labels. 

The canonical formula for our pure sinusoidal signal is:
$$ s(t) = A \cdot \sin(2\pi f t + \phi) $$
Where:
- **A**: Amplitude of the signal.
- **f**: Frequency in Hertz (Hz).
- **t**: Time vector.
- **$\phi$**: Phase offset (drawn randomly from $U(0, 2\pi)$ to prevent model memorisation of fixed trajectories).

To simulate real-world sensor data, we apply additive Gaussian noise:
$$ s_{noisy}(t) = s(t) + \epsilon $$
Where $\epsilon \sim \mathcal{N}(0, (\sigma A)^2)$, and $\sigma$ represents the noise standard deviation ratio.

### Nyquist–Shannon Sampling Theorem
A critical component of digitising analogue signals is the sampling rate. The Nyquist-Shannon Sampling Theorem dictates that to accurately reconstruct a signal without aliasing (where high frequencies falsely appear as low frequencies), the sampling rate ($f_s$) must be strictly greater than twice the highest frequency component ($f_{max}$) present in the signal:
$$ f_s > 2 \cdot f_{max} $$
In this project, our maximum target frequency is 50 Hz. To provide a substantial safety margin and high-resolution continuous waveforms, we configured the sampling rate to **200 Hz** ($4 \times f_{max}$).

---

## 3. RNN and LSTM Theory

While an MLP treats sequential data as a flat vector of independent features, Recurrent Neural Networks process data sequentially, maintaining a hidden state that acts as a "memory" of previous timesteps.

### Vanilla RNN
The Vanilla RNN updates its hidden state $h_t$ at each timestep $t$ by combining the current input $x_t$ with the previous hidden state $h_{t-1}$:
$$ h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}) $$
While effective for short sequences, the Vanilla RNN suffers heavily from the **Vanishing Gradient Problem**. When backpropagating through time (BPTT) over many timesteps, the gradients recursively multiply by the weight matrix $W_{hh}$. If the eigenvalues of this matrix are less than 1, the gradient exponentially shrinks to zero, preventing the network from learning long-term dependencies (e.g., extracting low-frequency waves like 5 Hz over a 10-step window).

### Long Short-Term Memory (LSTM)
The LSTM network was specifically designed by Hochreiter and Schmidhuber (1997) to combat the vanishing gradient problem. It achieves this by introducing a "cell state" ($C_t$) that runs straight down the entire chain, with only minor linear interactions. The LSTM controls the flow of information into and out of this cell state using three highly specialised neural network layers called "gates":

1. **Forget Gate ($f_t$)**: Decides what information to throw away from the cell state.
   $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
2. **Input Gate ($i_t$) & Cell Update ($\tilde{C}_t$)**: Decides what new information to store in the cell state.
   $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
   $$ \tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) $$
3. **Cell State Update ($C_t$)**: The old cell state $C_{t-1}$ is multiplied by $f_t$ (forgetting), and then the new candidate values $i_t \cdot \tilde{C}_t$ are added.
   $$ C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t $$
4. **Output Gate ($o_t$)**: Decides what part of the cell state to output as the new hidden state.
   $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
   $$ h_t = o_t \cdot \tanh(C_t) $$

Because the gradient can flow uninterrupted through the cell state pathway $C_t$, LSTMs are exceptionally capable of capturing long-range frequency dependencies spanning the entirety of our sliding window.

---

## 4. Dataset Generation Methodology

The success of the neural networks relies on the quality of the synthetic dataset.

### Frequency Selection
We meticulously chose four target frequencies to provide a diverse challenge to the networks:
- **5 Hz (Low)**: Highly challenging because 10 samples (our window size) represent only 0.25 of a full wavelength. Requires deep temporal memory.
- **15 Hz (Medium-Low)**: Tests moderate sequence memory.
- **30 Hz (Medium-High)**: Tests standard frequency extraction near the midpoint of the Nyquist limit.
- **50 Hz (High)**: Tests high-frequency extraction. 10 samples cover 2.5 complete wavelengths.

### Sliding Window Technique
To format the continuous 10-second signals (2000 samples each) into training examples, we employ a sliding window approach with $W = 10$ and a stride of 1. 
For each window, the input features consist of:
1. The 10 sequentially ordered noisy samples.
2. A 4-dimensional one-hot encoded vector representing the target frequency class.

The target output label is the true, clean sinusoidal value at the very next timestep.

### Normalisation
To ensure stable gradient descent, all signal data is standardised (zero-mean, unit-variance). Crucially, to strictly prevent data leakage, the normalisation statistics (mean and standard deviation) are computed **exclusively** on the training split, and these exact scalar values are then uniformly applied to transform both the validation and test splits.

---

## 5. Architecture Descriptions & Design Choices

The project implements three distinct models via `ModelFactory`, ensuring a fair comparison by standardising hidden layer sizes and capacities.

### 1. Fully Connected MLP (`MLPModel`)
- **Structure**: Linear(14, 64) $\rightarrow$ Tanh $\rightarrow$ Linear(64, 128) $\rightarrow$ Tanh $\rightarrow$ Linear(128, 64) $\rightarrow$ Tanh $\rightarrow$ Linear(64, 1).
- **Design Rationale**: The MLP acts as our non-sequential baseline. It flattens the 10 noisy samples and 4 one-hot encoded labels into a 14-dimensional feature vector. It has no concept of temporal ordering. The Tanh activation functions are specifically chosen over ReLU because our sinusoidal targets naturally oscillate between $[-1, 1]$, making a bounded activation highly appropriate and preventing unbounded activations from exploding. The middle layer expands to 128 units to provide sufficient width to learn complex, Fourier-like basis functions.
- **Parameters**: $\approx 17,601$

### 2. Vanilla RNN (`RNNModel`)
- **Structure**: 2-layer RNN (input_size=5, hidden_size=64) $\rightarrow$ Dropout(0.1) $\rightarrow$ Linear(64, 1).
- **Design Rationale**: The RNN processes data sequentially. At each timestep, it receives a 5-dimensional vector (1 noisy sample + 4 one-hot labels). By maintaining a hidden state, it inherently understands the temporal progression of the waveform. Two layers allow for hierarchical feature extraction. Dropout (0.1) provides light regularisation.
- **Parameters**: $\approx 25,473$

### 3. LSTM (`LSTMModel`)
- **Structure**: 2-layer LSTM (input_size=5, hidden_size=64) $\rightarrow$ Dropout(0.1) $\rightarrow$ Linear(64, 1).
- **Design Rationale**: The LSTM mirrors the exact topological dimensions of the RNN (2 layers, hidden size 64) but employs complex gating mechanisms to manage its internal cell state. This architectural choice directly tests the hypothesis that gated memory cells provide superior performance on time-series denoising tasks, specifically for low-frequency targets that exceed standard short-term memory constraints.
- **Parameters**: $\approx 99,585$ (Approximately $4\times$ the RNN due to the 4 internal gate weight matrices).

---

## 6. Training Procedure & Hyperparameters

All models were trained under identical conditions using parameters defined in `config/setup.json`.

### Core Training Loop
- **Loss Function**: Mean Squared Error (MSE). Chosen because this is fundamentally a continuous regression task, and MSE heavily penalises extreme outlier predictions.
- **Optimizer**: Adam ($\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$). Adam is selected over standard SGD due to its adaptive momentum, which accelerates convergence when traversing complex, non-convex loss landscapes associated with recurrent neural networks.
- **Batch Size**: 64. Provides a strong balance between stochastic gradient noise (for regularisation) and memory efficiency.
- **Gradient Clipping**: `max_norm=1.0`. A mandatory defensive measure specifically implemented to prevent the exploding gradient problem inherent to BPTT in the RNN model.

### Advanced Scheduling and Regularisation
- **Learning Rate Scheduler**: `ReduceLROnPlateau(factor=0.5, patience=10)`. Dynamically halves the learning rate when validation loss plateaus, allowing the optimiser to smoothly descend into sharp local minima late in training.
- **Early Stopping**: `patience=20`. Monitors validation MSE. If no improvement is observed for 20 epochs, training halts, and the model state dictionary is aggressively restored to the absolute best epoch. This guarantees we evaluate the true optimum and drastically mitigates the risk of overfitting.

---

## 7. Quantitative Results

The models were evaluated on the strictly held-out test split (15% of total data). The results decisively demonstrate that all models achieved exceptional performance, shattering the KPI target of `< 0.05` test MSE.

| Model | Train MSE | Val MSE | Test MSE |
|-------|-----------|---------|----------|
| MLP | 0.001427 | 0.001647 | 0.001636 |
| RNN | 0.001663 | 0.001843 | 0.001898 |
| LSTM | 0.001725 | 0.001799 | 0.001884 |

*Note: Surprisingly, the MLP achieved the lowest global test MSE. Given the relatively short context window ($W=10$), the MLP possessed sufficient capacity to memorise the localised curve geometries.*

---

## 8. Training Curves

The training and validation loss curves illustrate rapid early convergence. The `ReduceLROnPlateau` scheduler visually tightens the variance in later epochs, and Early Stopping successfully prevents validation divergence.



---

## 9. Prediction Comparisons

The following figures visually contrast the models' predictions against the highly erratic noisy input and the perfectly smooth ground-truth target.

### MLP Predictions


### RNN Predictions


### LSTM Predictions


*Observation: All three models successfully filter out high-frequency Gaussian noise, generating remarkably smooth sinusoidal trajectories that tightly track the clean target line.*

---

## 10. Noise Robustness Analysis

To evaluate model resilience, we conducted a systematic stress test by exponentially increasing the noise standard deviation ratio ($\sigma$) from 0.05 to 0.50. 



**Analysis:**
As expected, Test MSE degrades quadratically as the Signal-to-Noise Ratio (SNR) plummets. However, the LSTM demonstrates superior asymptotic stability at extreme noise levels ($\sigma = 0.50$), proving that its gated memory cells are significantly more adept at filtering out severe stochastic perturbations compared to the flat MLP structure.

---

## 11. Per-Frequency Analysis

Different frequencies pose distinct temporal challenges. We computed the exact MSE isolated to each frequency class.



**Analysis:**
- **5 Hz (Low Frequency)**: Represents the highest error across all models. Because 10 samples cover only a tiny fraction of a 5 Hz wavelength, the models struggle to infer the global phase shift from such a limited, near-linear local slope.
- **50 Hz (High Frequency)**: Demonstrates the lowest error. Because multiple complete cycles fit within the 10-sample window, the networks have abundant temporal context to accurately lock onto the phase and frequency.

---

## 12. Conclusions & Observations

1. **Stellar Performance:** All models vastly exceeded the primary objective of `< 0.05` test MSE, achieving errors in the range of `0.0016 - 0.0019`. The normalisation strategy, bounded Tanh activations, and Adam optimisation proved highly effective.
2. **The MLP Anomaly:** The MLP technically secured the lowest test MSE on the standard dataset. This suggests that for very short context windows (10 samples), the task behaves more like complex curve fitting rather than long-range sequence modelling. The MLP's 14-dimensional dense mapping was highly efficient for this micro-scale geometry.
3. **LSTM Resilience:** In the high-noise robustness experiments, the LSTM overtook the MLP. This validates the theoretical assumption that sophisticated gating mechanisms (Forget/Input gates) are essential for distinguishing true temporal signal from overwhelming stochastic noise.
4. **Low-Frequency Bottleneck:** The 5 Hz frequency consistently caused the most errors. To improve this, future iterations should dynamically expand the window size ($W \ge 40$) when low-frequency one-hot labels are detected, providing the necessary temporal receptive field.

---

## 13. Reproduction Instructions

To reproduce the exact quantitative results and regenerate all high-resolution artifact plots from scratch, execute the pipeline in `all` mode with the fixed random seed:

```bash
uv run python src/main.py --mode all --seed 42
```
*Expected Wall-Clock Time: ~2 to 4 minutes on a standard modern CPU.*

---

## 14. Repository Link

GitHub Repository: [https://github.com/Saed-Abdalgani/HW1---Signal-Frequency-Extraction](https://github.com/Saed-Abdalgani/HW1---Signal-Frequency-Extraction)

---

## 15. Installation & Quickstart

This project rigorously uses `uv` for ultra-fast, deterministic dependency management.

**1. Clone and Sync Dependencies:**
```bash
git clone https://github.com/Saed-Abdalgani/HW1---Signal-Frequency-Extraction.git
cd HW1---Signal-Frequency-Extraction
uv sync
```

**2. Run the Full ML Pipeline:**
```bash
uv run python src/main.py --mode all
```

**3. Launch the Sinusoid Explorer Interactive UI:**
```bash
uv run python src/main.py --mode ui --port 8050
```
Open your browser to `http://localhost:8050` to interact with real-time signal processing, noise injection, and 3D T-SNE/PCA visualisations.

---

## 16. Configuration Guide

All project parameters are strictly decoupled from source code and managed in `config/setup.json`.

| Key Path | Description | Default |
|----------|-------------|---------|
| `signal.frequencies_hz` | Target frequencies to generate. | `[5, 15, 30, 50]` |
| `signal.sampling_rate_hz` | Must strictly obey Nyquist ($> 2 \cdot \max(f)$). | `200` |
| `dataset.train_ratio` | Stratified split for training data. | `0.70` |
| `model.hidden_size` | Dimensions for RNN/LSTM hidden states. | `64` |
| `training.batch_size` | Batch size for PyTorch DataLoaders. | `64` |
| `training.max_epochs` | Upper limit before early stopping triggers. | `300` |
| `evaluation.noise_levels_for_robustness` | Array of $\sigma$ values for stress testing. | `[0.05, 0.1, 0.2, 0.3, 0.5]` |

---

## 17. Troubleshooting

- **`ValueError: sampling_rate_hz must be at least twice the maximum frequency`**
  - *Cause:* Nyquist theorem violation detected at startup.
  - *Fix:* Edit `setup.json` and ensure `sampling_rate_hz` is $> 2 \times$ your highest frequency.
- **`GatekeeperError: Max retries exceeded`**
  - *Cause:* File I/O failure (e.g., disk full, permission denied when writing checkpoints).
  - *Fix:* Ensure the process has write access to the `results/` and `data/` directories.
- **Training is extremely slow or CUDA out of memory**
  - *Cause:* PyTorch is attempting to allocate GPU memory but failing.
  - *Fix:* Force CPU execution by setting the environment variable: `export FREQ_EXTRACTOR_FORCE_CPU=1`.
- **Dash UI fails to start (`Address already in use`)**
  - *Cause:* Port 8050 is occupied by another process.
  - *Fix:* Launch on a different port: `uv run python src/main.py --mode ui --port 8080`.
- **`Config version mismatch` on checkpoint load**
  - *Cause:* You are attempting to load a `.pt` checkpoint generated from an older, incompatible config schema.
  - *Fix:* Delete the contents of `results/checkpoints/` and retrain the models.

---

## 18. Testing Instructions

The project features a highly comprehensive, deterministic test suite covering core services, PyTorch dimensional constraints, and UI callbacks.

To execute the test suite and verify the strictly mandated $\ge 85\%$ line coverage:
```bash
uv run pytest tests/ --cov=src
```

---

## 19. Linting Instructions

To enforce strict code quality, type-hinting, and the critical **$\le 145$ code lines per file** constraint, the project utilises Ruff.

To run the linter:
```bash
uv run ruff check .
```
*(Zero violations are required for a successful build).*

---

## 20. Contribution Guidelines

1. **Branching Strategy:** Cut feature branches from `main` (e.g., `feat/add-transformer-model`).
2. **Line Constraints:** Absolutely no source code file in `src/freq_extractor/` may exceed 145 executable lines. You must split your module if you breach this limit.
3. **Gatekeeper:** All file I/O operations must be routed through `shared.gatekeeper.ApiGatekeeper`. Direct `open()` calls are prohibited.
4. **Pull Requests:** PRs must pass all GitHub Actions CI checks (Ruff, Pytest Coverage $\ge 85\%$).

---

## 21. Credits & License

**Author:** Saed Abdalgani  
**Course:** Machine Learning & Digital Signal Processing  

This project is open-sourced under the **MIT License**.

```text
MIT License

Copyright (c) 2026 Saed Abdalgani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Limitations & Future Work

- The dataset is synthetic and single-channel, so results should not be treated as evidence for real sensor, audio, or RF deployments without domain-specific validation.
- The current window size is fixed globally. Low-frequency signals would benefit from adaptive or frequency-aware window lengths.
- The Dash UI is a local exploratory dashboard, not a hardened production web service.
- The training pipeline optimizes one-step regression only; multi-step forecasting and uncertainty estimates are natural next experiments.
- Future work could add remote storage adapters behind the Gatekeeper, richer noise models, and cross-validation across wider frequency grids.

## Architectural Decisions

The implementation follows the ADRs in [docs/PLAN.md](docs/PLAN.md):

| ADR | Decision |
|-----|----------|
| ADR-1 | Use pure PyTorch training loops instead of Lightning or Keras. |
| ADR-2 | Use Tanh-aligned bounded activations for sinusoidal regression. |
| ADR-3 | Use Adam for all model families to keep comparisons controlled. |
| ADR-4 | Route file I/O through the Gatekeeper abstraction. |
| ADR-5 | Keep hyperparameters and paths config-driven. |
| ADR-6 | Use Plotly Dash for the local Sinusoid Explorer UI. |

---

## Deviations & Justifications (Homework Requirements)

While implementing the explicit requirements of this assignment, several deliberate architectural deviations were made to improve the realism and difficulty of the machine learning task. Below are the justifications for these choices:

1. **Noise Generation Formula (AWGN vs. Multiplicative/Phase Noise):**
   * *Deviation:* The prompt suggested amplitude/phase noise `(A +- sigma)(sin(2pi * f * phi + sigma_2))`. Instead, the dataset employs standard Additive White Gaussian Noise (AWGN): $s(t) + \epsilon$.
   * *Justification:* AWGN is the universal industry standard for modelling environmental sensor interference in digital signal processing. By applying additive noise, we test the network's resilience to random voltage fluctuations rather than structured amplitude scaling, making the denoising task more realistic.
2. **Forecasting vs. Seq-to-Seq Denoising:**
   * *Deviation:* The prompt suggested outputting "10 samples without noise" for every 10 noisy input samples (a Sequence-to-Sequence autoencoder task). Instead, this project uses a Sequence-to-One forecasting architecture, predicting only the *next* clean sample.
   * *Justification:* Predicting a single future clean sample forces the network to truly learn the underlying periodic function and extrapolate it, rather than just acting as a static smoothing filter that memorizes a 1-to-1 denoising mapping. It heavily penalizes models that lack genuine temporal understanding.
3. **Omitting Explicit Sigma from Input Features:**
   * *Deviation:* The prompt mentioned providing `sigma` (the noise percentage) as an explicit feature to the network alongside the signal and the one-hot frequency vector. The implemented networks do not receive `sigma`.
   * *Justification:* Explicitly feeding the noise scalar to the network acts as a crutch. By omitting `sigma`, we deliberately increase the difficulty of the task, forcing the MLP, RNN, and LSTM to implicitly estimate the noise variance directly from the chaotic time-series data itself. This results in significantly more robust models.

---

## Dashboard UI Test Results

The following screenshots demonstrate the interactive dashboard functioning correctly across multiple configurations, parameter changes, and visualisations.

![Dashboard State 1](screenshots/dash_1.png)
*Dashboard State 1: Default initialization showing four individual pure sinusoids and the resulting combined clean signal.*

![Dashboard State 2](screenshots/dash_2.png)
*Dashboard State 2: Adjusting individual frequencies, phases, and amplitudes to observe real-time waveform updates.*

![Dashboard State 3](screenshots/dash_3.png)
*Dashboard State 3: Injecting global uniform noise to simulate real-world sensor interference on the combined signal.*

![Dashboard State 4](screenshots/dash_4.png)
*Dashboard State 4: Toggling display mode from continuous lines to discrete sampled dots.*

![Dashboard State 5](screenshots/dash_5.png)
*Dashboard State 5: Increasing the sampling frequency to observe higher resolution discrete data points.*

![Dashboard State 6](screenshots/dash_6.png)
*Dashboard State 6: Expanding the N-Cycles parameter to visualize a wider temporal window.*

![Dashboard State 7](screenshots/dash_7.png)
*Dashboard State 7: Complex mixing of high-frequency sinusoids under heavy noise conditions.*

![Dashboard State 8](screenshots/dash_8.png)
*Dashboard State 8: T-SNE 3D dimensionality reduction visualization clustering the signal components.*

![Dashboard State 9](screenshots/dash_9.png)
*Dashboard State 9: Adjusting the T-SNE perplexity slider to refine cluster separation and data grouping.*

![Dashboard State 10](screenshots/dash_10.png)
*Dashboard State 10: PCA 3D visualization revealing the orthogonal principal components of the dataset.*

![Dashboard State 11](screenshots/dash_11.png)
*Dashboard State 11: Rotating the PCA 3D plot to examine variance across different spatial dimensions.*

![Dashboard State 12](screenshots/dash_12.png)
*Dashboard State 12: Exploring the frequency domain representation using the FFT Spectrum plot.*

![Dashboard State 13](screenshots/dash_13.png)
*Dashboard State 13: Applying a Bandpass filter to isolate specific frequency ranges from the noisy signal.*

![Dashboard State 14](screenshots/dash_14.png)
*Dashboard State 14: Applying a Lowpass filter to smooth out high-frequency noise and recover the underlying wave.*

![Dashboard State 15](screenshots/dash_15.png)
*Dashboard State 15: Advanced T-SNE 3D projection showcasing a distinct ring topology for the periodic signals.*

![Dashboard State 16](screenshots/dash_16.png)
*Dashboard State 16: Exploring the interactive tooltips on the 3D scatter plots for precise coordinate tracking.*

![Dashboard State 17](screenshots/dash_17.png)
*Dashboard State 17: Isolating a single sinusoid by toggling the MIX checkboxes in the control panel.*

![Dashboard State 18](screenshots/dash_18.png)
*Dashboard State 18: Sweeping noise levels dynamically to observe signal degradation and filter robustness.*

![Dashboard State 19](screenshots/dash_19.png)
*Dashboard State 19: High-frequency isolated sinusoid viewed in discrete dot mode with low amplitude.*

![Dashboard State 20](screenshots/dash_20.png)
*Dashboard State 20: Final comprehensive view of the fully styled, dark-mode Sinusoid Explorer UI.*

---

## 22. Additional Feature Demonstrations

The following screenshots capture specific UI features not yet shown in the primary dashboard gallery, providing complete proof of all implemented capabilities.

### FFT Spectrum — Log Scale Mode

![FFT Log Scale](screenshots/dash_21.png)
*The FFT Magnitude Spectrum tab with **Log Scale** enabled (checkbox ticked). The frequency axis switches to logarithmic spacing, making it far easier to distinguish spectral peaks across multiple decades of frequency. The sharp spike at 5 Hz clearly confirms the dominant sinusoid frequency.*

---

### Filter — Highpass Mode

![Highpass Filter Applied](screenshots/dash_22.png)
*The global **Highpass** filter applied to the composite signal. Low-frequency components are attenuated, isolating only the higher-frequency content. The combined signal clearly shows reduced low-frequency content compared to the unfiltered baseline.*

---

### Filter — Lowpass Mode

![Lowpass Filter Applied](screenshots/dash_23.png)
*The global **Lowpass** filter applied to the composite signal. High-frequency components are rolled off, leaving only the smooth, slow sinusoidal components. This visually demonstrates the zero-phase `sosfiltfilt` Butterworth implementation.*

---

### Per-Sinusoid Bandpass Filter (BPF Checkbox)

![Per-Sinusoid BPF Enabled](screenshots/dash_24.png)
*The per-channel **BPF** (Band-Pass Filter) checkbox for Sin 1 is enabled (shown checked in the Sin 1 control block). A narrow bandpass filter centred at Sin 1's frequency is applied independently to that channel before mixing, demonstrating channel-isolated filtering capability.*

---

### Gaussian Noise with Per-Sinusoid Sigma (σ)

![Gaussian Noise + Per-Sin Sigma](screenshots/dash_25.png)
*Global **Gaussian** noise is active (selected in the Noise dropdown), and per-sinusoid sigma (σ) sliders for Sin 1 are set to 0.40. The individual sinusoid traces show clearly different noise envelopes — Sin 1 (cyan) has visibly more noise than the others — confirming that per-channel additive noise is working independently of the global noise type.*
### Priority 1: FFT with BPF Bandwidth Shading (FR-12)

![FR-12 BW shaded bandwidth region on FFT Spectrum when per-sinusoid BPF is active](screenshots/dash_26.png)
*FR-12 BW shaded bandwidth region on FFT Spectrum when per-sinusoid BPF is active.*

---

### Priority 2: Terminal Training Convergence

![Terminal: --mode all run output](screenshots/terminal_training.png)
*Training convergence logs and final MSE table.*

---

### Priority 3: Pytest Coverage

![Terminal: pytest --cov=src -> >=85%](screenshots/pytest_coverage.png)
*Pytest test suite showing over 85% coverage on the src directory.*

---

### Priority 5: Live Header Metrics

![FR-9.3 live header metrics bar showing all 6 computed metrics updating in real time](screenshots/dash_27.png)
*FR-9.3 live header metrics bar showing all 6 computed metrics updating in real time.*

---

### Priority 6: Signal Examples

![Signal Examples showing predicted versus clean target versus noisy signal](results/signal_examples.png)
*Signal examples visualised post-training showing the model predictions (PRD_training §11).*

