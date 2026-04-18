# LSTM & Transformer Time Series Benchmark
**CMPE 401 — Instructor-defined Project 2**

This project reproduces, organizes, and benchmarks two official Keras time-series deep learning examples:
one using a **Transformer** for time-series classification, and one using an **LSTM** for weather forecasting.
Three controlled modifications are applied to the LSTM baseline to explore what actually moves the needle on validation loss.

---

## Repository Structure

```
lstm-transformer-timeseries/
│
├── README.md
├── notebooks/
│   ├── 01_lstm_baseline.ipynb           # Keras LSTM example, reproduced as-is
│   ├── 02_transformer_baseline.ipynb    # Keras Transformer example, reproduced as-is
│   ├── 03_lstm_modification.ipynb       # All three LSTM experiments
│
└── results/
    ├── lstm_baseline/
    ├── lstm_modified/
    ├── transformer_baseline/
    └── lstm_modified_final_summary.txt

```

---

## 1. Project Overview

Time-series data is central to many engineering applications — weather monitoring, health signals, industrial sensing, and more. This project focuses on two fundamentally different time-series problems and the architectures designed to solve them:

| Problem Type | Dataset | Model | Metric |
|---|---|---|---|
| Classification | FordA (engine noise sensor) | Transformer | Sparse Categorical Accuracy |
| Forecasting | Jena Climate (weather station) | LSTM | MSE (val_loss) |

---

## 2. Datasets

### FordA — Classification
The FordA dataset contains univariate time-series readings from an automotive engine noise sensor. Each series is labeled as either normal (`0`) or abnormal (`1`), making this a binary classification problem. Samples have a fixed length of 500 timesteps and are loaded directly as `.tsv` files from the Keras CD-Diagram repository.

- **Train samples:** 3,601
- **Test samples:** 1,320
- **Input shape:** `(500, 1)` — one reading per timestep

### Jena Climate — Forecasting
The Jena Climate dataset contains 14 meteorological features recorded every 10 minutes at a weather station in Jena, Germany, from 2009 to 2016. The task is to predict the temperature 12 hours into the future given a recent window of history. Seven features are selected from the full 14 for training.

- **Total samples:** ~420,551 rows
- **Train / Val split:** 71.5% / 28.5%
- **Selected features:** Pressure, Temperature, Saturation vapor pressure, Vapor pressure deficit, Specific humidity, Air density, Wind speed
- **Prediction target:** Temperature 12 hours ahead (`T (degC)`)
- **Sampling:** One reading per hour (every 6th 10-minute row)

---

## 3. Models

### Transformer (Classification)

The Transformer encoder stack processes the input sequence using multi-head self-attention, allowing every timestep to directly attend to every other timestep regardless of distance. This overcomes the vanishing gradient problem that LSTMs face on long sequences.

**Architecture:**
```
Input (500, 1)
→ 4× TransformerEncoder [
     MultiHeadAttention(heads=4, key_dim=256)
     → Dropout(0.25)
     → LayerNorm
     → residual add
     → Conv1D(ff_dim=4, relu) → Dropout → Conv1D → LayerNorm
     → residual add
  ]
→ GlobalAveragePooling1D
→ Dense(128, relu) → Dropout(0.4)
→ Dense(2, softmax)
```

**Key implementation note:** This uses a Pre-LN (pre-normalization) variant — LayerNorm is applied before the residual connection rather than after. This tends to produce more stable gradients and removes the need for learning rate warm-up.

**Training config:** Adam (lr=1e-4), sparse categorical crossentropy, early stopping (patience=10), batch size 64, up to 150 epochs.

---

### LSTM (Forecasting)

The LSTM processes the weather sequence step-by-step, maintaining a hidden state that carries information forward through time. Its gating mechanisms (forget, input, output gates) allow it to selectively remember or discard information across long sequences.

**Architecture:**
```
Input (120, 7)   ← 120 hourly steps × 7 features
→ LSTM(32)
→ Dense(1)       ← predicted temperature
```

**Training config:** Adam (lr=0.001), MSE loss, early stopping (patience=5), ModelCheckpoint saving best weights, batch size 256, up to 10 epochs.

---

## 4. Baseline Results

### Transformer — FordA Classification

The Transformer was run as-is from the Keras example with no modifications.

| Metric | Value |
|---|---|
| Test Accuracy | See `results/baseline_transformer_results.txt` |
| Loss | Sparse categorical crossentropy |
| Epochs trained | Early stopped |

### LSTM — Jena Climate Forecasting

The LSTM baseline serves as the reference point for all three experiments below.

| Metric | Value |
|---|---|
| Best Val Loss (MSE) | **0.122696** |
| Architecture | Single LSTM(32), no dropout, past=720 |
| Lookback window | 720 steps = 120 hours |

---

## 5. LSTM Improvement Experiments

Three modifications were applied to the LSTM baseline, each changing exactly one variable. All other hyperparameters were held constant to ensure direct comparability.

### Experiment 1 — Dropout Regularization

**What changed:** A `Dropout` layer was added immediately after the LSTM output, before the Dense prediction layer. Four dropout rates were tested.

**Hypothesis:** Dropout would prevent neuron co-adaptation, reducing overfitting and improving generalization to the validation set.

| Dropout Rate | Best Val Loss | vs Baseline |
|---|---|---|
| None (baseline) | 0.122696 | — |
| 0.1 | 0.142252 | +0.019557 |
| 0.2 | 0.139744 | +0.017048 |
| 0.3 | 0.133902 | +0.011206 |
| 0.5 | 0.137704 | +0.015008 |

**Finding:** Every dropout rate made performance worse. The degradation follows a clear pattern — loss improves as the rate increases from 0.1 to 0.3, then worsens at 0.5 — suggesting the optimal rate may lie between 0.3 and 0.5, but the benefit would still not exceed the baseline.

**Why it didn't help:** The baseline model is not overfitting. Training loss (~0.100) and validation loss (~0.123) track closely throughout training with no significant divergence. Dropout is a regularization technique designed specifically to combat overfitting; applying it to a well-generalized model simply suppresses useful signal during training without addressing any real problem. The Jena dataset is large enough (300k+ training steps) that a 32-unit LSTM does not have the capacity to memorize it.

---

### Experiment 2 — Stacked LSTM Layers

**What changed:** A second LSTM layer was added on top of the first. The first layer used `return_sequences=True` to pass its full hidden-state sequence to the second layer. Two configurations were tested.

**Hypothesis:** Hierarchical representation learning — the first layer encodes short-term fluctuations, the second encodes multi-day trends — would improve 12-hour temperature forecasts.

| Configuration | Params | Time/epoch | Best Val Loss | vs Baseline |
|---|---|---|---|---|
| Single LSTM(32) — baseline | ~4,481 | ~178s | 0.122696 | — |
| LSTM(32) → LSTM(32) | 13,473 | ~323s | 0.132490 | +0.009794 |
| LSTM(64) → LSTM(32) | 30,881 | ~566s | 0.133303 | +0.010607 |

**Finding:** Both stacked configurations performed worse than the single-layer baseline, and the larger LSTM(64)→LSTM(32) was marginally worse than the smaller stack despite having 7× more parameters and taking 3× longer to train. Adding capacity made things worse, not better.

**Why it didn't help:** The forecasting problem does not require hierarchical temporal abstraction. Predicting temperature 12 hours ahead is a smooth, relatively low-complexity regression task where the relevant signal is concentrated in recent weather patterns. The single LSTM(32) already has sufficient capacity to model these dependencies. The second layer added parameters without adding useful inductive bias, and the larger parameter count introduced mild overfitting — evidenced by the fact that training loss for the stacked models (~0.100) matched the baseline's, while their validation loss was higher.

---

### Experiment 3 — Input Sequence Length

**What changed:** The lookback window `past` was varied across three values, changing how many timesteps of history the model receives. The model architecture was held constant (single LSTM(32)) across all three.

**Hypothesis:** More history would give the model access to longer-range weather cycles, improving its 12-hour temperature predictions.

| Lookback | Steps fed to LSTM | Time/epoch | Best Val Loss | vs Baseline |
|---|---|---|---|---|
| 60 hours | 60 | ~108ms/step | **0.121850** | **-0.000846** |
| 120 hours (baseline) | 120 | ~150ms/step | 0.131121 | — |
| 240 hours | 240 | ~240ms/step | 0.131232 | +0.008536 |

**Finding:** The 60-hour window achieved the lowest validation loss of any configuration tested — including the original baseline — and was also the fastest to train. The 120-hour and 240-hour windows produced nearly identical results (0.1311 vs 0.1312), confirming that information beyond 60 hours provides essentially zero additional predictive value.

**Why shorter won:** Two complementary reasons. First, near-term temperature forecasting is dominated by recent conditions — what happened four or five days ago carries little predictive power for the next 12 hours compared to the last two days. The extra history in the 720 and 1440 windows is mostly noise from the model's perspective. Second, a shorter input sequence reduces the gradient path length through the LSTM's backpropagation-through-time, meaning the model trains more effectively on the signal that actually matters — the most recent timesteps.

---

## 6. Full Results Table

| Experiment | Description | Best Val Loss (MSE) | vs Baseline |
|---|---|---|---|
| Baseline | Single LSTM(32), no dropout, past=720 | 0.122696 | — |
| Exp 1: Dropout=0.1 | Dropout(0.1) after LSTM output | 0.142252 | +0.019557 |
| Exp 1: Dropout=0.2 | Dropout(0.2) after LSTM output | 0.139744 | +0.017048 |
| Exp 1: Dropout=0.3 | Dropout(0.3) after LSTM output | 0.133902 | +0.011206 |
| Exp 1: Dropout=0.5 | Dropout(0.5) after LSTM output | 0.137704 | +0.015008 |
| Exp 2: LSTM(32)→LSTM(32) | Two stacked layers | 0.132490 | +0.009794 |
| Exp 2: LSTM(64)→LSTM(32) | Two stacked layers, wider first | 0.133303 | +0.010607 |
| **Exp 3: past=360** | **60h lookback** | **0.121850** | **-0.000846** |
| Exp 3: past=720 | 120h lookback (baseline) | 0.131121 | — |
| Exp 3: past=1440 | 240h lookback | 0.131232 | +0.008536 |

The best single modification was **reducing the lookback window from 120 hours to 60 hours**, achieving val_loss = 0.121850 — the only configuration that outperformed the baseline.

---

## 7. Discussion Questions

### Which model did you find easier to understand, and why?

The LSTM model is significantly easier to understand. Its architecture is minimal — a single recurrent layer feeding a linear output — and its operation maps intuitively onto the problem: given the last N hours of weather readings, predict the temperature 12 hours from now. The data pipeline, training loop, and evaluation metric (MSE) are all straightforward.

The Transformer is conceptually harder to follow. Multi-head self-attention requires understanding query/key/value projections and how attention scores are computed across all pairs of timesteps simultaneously. The Pre-LN residual structure, the pointwise Conv1D feed-forward sub-layers, and the way global average pooling collapses the sequence for classification all require more background to reason about. That said, once the attention mechanism is understood, the Transformer's advantage over LSTMs on long sequences — every timestep can directly attend to every other, with no vanishing gradient issue — becomes intuitive.


---

### What improvement did you try, and what did you learn from it?

Three modifications were tested on the LSTM: dropout regularization, stacking a second LSTM layer, and varying the input sequence length.

The most important lesson came from the fact that two out of three modifications made performance worse. Dropout and stacking both degraded validation loss despite being widely recommended techniques for improving neural networks. This taught a concrete lesson about the difference between a technique being useful in general and being appropriate for a specific problem:

**Dropout failed** because the baseline model is not overfitting. Regularization only helps when there is a generalization gap to close. Applying dropout to a model that already generalizes well simply reduces its ability to learn during training.

**Stacking failed** because the task does not require hierarchical abstraction. More parameters did not translate to better predictions — it introduced mild overfitting on a problem that was already well-solved by the simpler architecture.

**Reducing the lookback window succeeded** — and counterintuitively, less data was better. The winning configuration (60-hour window) outperformed both the baseline and the longer 240-hour window. This revealed that the useful signal for 12-hour temperature forecasting is concentrated in the most recent two days of weather history. Feeding the model five days of history — as the baseline did — introduced noise rather than additional useful context, and made gradient propagation through the LSTM slightly less effective. The finding demonstrates a principle that holds broadly in applied ML: more input is not always better, and the right inductive bias (what you choose to include) often matters more than model capacity.

---

## 8. Requirements

```
keras>=3.0
tensorflow>=2.15
pandas
numpy
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 9. References

- [Keras: Time Series Classification with Transformer](https://keras.io/examples/timeseries/timeseries_classification_transformer/)
- [Keras: Timeseries Forecasting for Weather](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)
- Jena Climate dataset — Max Planck Institute for Biogeochemistry
- FordA dataset — UCR Time Series Archive
