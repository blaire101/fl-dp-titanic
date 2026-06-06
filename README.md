# Federated Learning with Differential Privacy
### Non-IID Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project implements **FedAvg + DP-SGD** from scratch to study the privacy-utility tradeoff under Non-IID federated learning settings, using the Titanic dataset.

Passengers are split by cabin class (Pclass) into 3 federated clients with significantly different survival rate distributions — a natural Non-IID scenario. Differential privacy is applied via Gaussian noise injection on gradients, with privacy budgets ε∈{5, 1} tested.

**Key finding:** ε=5 (σ=0.48) achieves a practical privacy-utility balance. ε=1 (σ=2.42) causes training collapse as noise magnitude exceeds the gradient signal.

---

## Results

| Method | Accuracy | F1-Score | ε | σ |
|--------|----------|----------|---|---|
| Centralized (upper bound) | 0.8034 | 0.7407 | N/A | — |
| FedAvg (No DP) | 0.7921 | 0.7218 | ∞ | 0 |
| DP-FedAvg (ε=5) | 0.6966 | 0.5500 | 5 | 0.4845 |
| DP-FedAvg (ε=1) | 0.4326 | 0.3567 | 1 | 2.4224 |

> All results evaluated on the same held-out test set (178 samples, 20% split, seed=42).
> C=0.5, δ=1e-5.

---

## Non-IID Setup

Titanic passengers split by Pclass into 3 clients with significantly different label distributions:

| Client | Pclass | Samples | Survival Rate |
|--------|--------|---------|---------------|
| Client 1 | 1st class | 172 | 63.4% |
| Client 2 | 2nd class | 151 | 45.0% |
| Client 3 | 3rd class | 386 | 24.9% |

Survival rate spans from 63.4% to 24.9% — a ~40pp gap — making this a challenging Non-IID federated setting where client gradients point in opposing directions.

---

## Model Architecture

```
Input (6 features)
    │
    ▼
Dense(32, ReLU)
    │
    ▼
Dense(16, ReLU)
    │
    ▼
Dense(1, Sigmoid)
    │
    ▼
Output: survival probability (≥0.5 → survived)
```

**Features:** Pclass, Sex, Age, SibSp, Parch, Fare (StandardScaler normalised)

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=0.001 |
| FL Rounds | 45 |
| Centralized epochs | 100, batch_size=32 |
| Clipping norm C | 0.5 |
| Privacy δ | 1e-5 |
| Privacy ε | 5 and 1 |
| Train/Test split | 80/20, seed=42 |

---

## DP-SGD Implementation

Noise standard deviation σ is derived from the Gaussian mechanism:

```
σ = √(2 · ln(1.25/δ)) · C / ε
```

| ε | σ | Effect |
|---|---|--------|
| 5 | 0.4845 | Moderate noise, model still converges |
| 1 | 2.4224 | Noise ~5× gradient magnitude, training collapses |

**Core training loop (per round, per client):**
1. Compute gradients via `tf.GradientTape`
2. Clip gradients: `tf.clip_by_norm(g, C=0.5)`
3. Add Gaussian noise: `g += N(0, σ²I)`
4. Federated aggregation: equal-weight average across 3 clients
5. Update global model via `optimizer.apply_gradients()`

---

## Repository Structure

```
fl-dp-titanic/
├── README.md
├── requirements.txt
├── notebook/
│   └── federated_dp_titanic.ipynb   # Full experiment notebook
├── src/
│   ├── fedavg.py                    # FedAvg training engine
│   ├── dp_sgd.py                    # DP-SGD Gaussian mechanism
│   └── evaluate.py                  # Evaluation metrics
├── figures/
│   ├── noniid_distribution.png      # Non-IID client distribution
│   ├── convergence_curve.png        # Accuracy vs rounds
│   ├── performance_compare.png      # Bar chart comparison
│   └── privacy_tradeoff.png         # ε vs Accuracy/F1
└── results/
    └── summary.csv                  # Experiment results
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/fl-dp-titanic.git
cd fl-dp-titanic
pip install -r requirements.txt
jupyter notebook notebook/federated_dp_titanic.ipynb
```

---

## References

1. McMahan H B, et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS, 2017.
2. Abadi M, et al. *Deep Learning with Differential Privacy.* ACM CCS, 2016.
3. Zhao Y, et al. *Federated Learning with Non-IID Data.* arXiv:1806.00582, 2018.
4. Dwork C, Roth A. *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in TCS, 2014.
5. Zhu L, et al. *Deep Leakage from Gradients.* NeurIPS, 2019.

---

## Author

**Chen Libin** | NAU Singapore | Data Engineer  
Course Project — Federated Learning with Differential Privacy
