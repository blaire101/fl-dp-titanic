# Logistic Regression vs MLP — From Linear to Non-linear Classification

A conceptual guide to two foundational classification models,
illustrated with the Titanic survival prediction task.

---

## 1. The Core Problem

Given a passenger's features (age, sex, cabin class, fare…),
predict whether they survived (1) or died (0).

This is a **binary classification** problem.
Both Logistic Regression and MLP produce a probability P(survived) ∈ [0, 1].
The decision threshold is 0.5 by default.

---

## 2. Logistic Regression (LR)

### How it works

LR draws a **single straight line** (or hyperplane in higher dimensions)
to separate the two classes.

```
Step 1 — Linear combination:
  z = w₁·Pclass + w₂·Sex + w₃·Age + w₄·SibSp + w₅·Parch + w₆·Fare + b

Step 2 — Sigmoid squash:
  P(survived) = σ(z) = 1 / (1 + e⁻ᶻ)
```

The sigmoid function converts any real number to a probability:

```
z = -5  →  σ(z) ≈ 0.007   (almost certain to die)
z =  0  →  σ(z) = 0.500   (50/50)
z = +5  →  σ(z) ≈ 0.993   (almost certain to survive)
```

### Visual intuition

```
  High survival
  probability
        │          ●  ●  ● ← survived
        │       ● /
  0.5 ──┼──────/──────────── decision boundary
        │   / ●
        │  / ○  ○  ○ ← died
  Low   │
        └──────────────────
             Feature value (e.g. Fare)
```

LR can only draw one straight line. If the true boundary is curved or twisted,
LR will always under-fit.

### Strengths & weaknesses

| ✓ Strengths | ✗ Weaknesses |
|-------------|-------------|
| Fast to train | Assumes linear separability |
| Highly interpretable (each weight has a clear meaning) | Cannot capture feature interactions (e.g. "female AND 1st class") |
| Rarely overfits on small data | Fixed model capacity — can't improve with more data |

---

## 3. Multi-Layer Perceptron (MLP)

### How it works

MLP stacks multiple layers of neurons.
Each layer learns a **non-linear transformation** of the previous layer's output.
By stacking layers, MLP can learn arbitrarily complex decision boundaries.

```
Architecture used in this project:

  Input (6)  →  Hidden₁ (32, ReLU)  →  Hidden₂ (16, ReLU)  →  Output (1, Sigmoid)
```

### Layer-by-layer

```
Layer 1 — Raw features enter:
  [Pclass=3, Sex=0, Age=25, SibSp=0, Parch=0, Fare=7.9]

Layer 2 — Hidden layer 1 (32 neurons, ReLU):
  Each neuron computes:  a = max(0,  w·x + b)
  ReLU keeps positive values, zeros out negatives.
  32 neurons learn 32 different "detectors"
  (e.g. one might learn "low fare AND male → likely died")

Layer 3 — Hidden layer 2 (16 neurons, ReLU):
  Combines the 32 detectors into 16 higher-level patterns.

Layer 4 — Output (1 neuron, Sigmoid):
  Combines all patterns into a single survival probability.
```

### ReLU activation

```
         output
           │     /
           │    /
           │   /
     ──────┼──/──────── input
           │/
           0

ReLU(x) = max(0, x)

Why ReLU?
• Simple and fast to compute
• Avoids the vanishing gradient problem of sigmoid in hidden layers
• Introduces non-linearity — without it, stacking layers would still
  just be one big linear function
```

### Visual intuition

```
LR — one straight boundary:          MLP — curved, complex boundary:

  ●  ●  │  ○  ○                        ●  ●  ╭───╮  ○
  ●  ●  │  ○  ○                        ●  ╭──╯   ╰──╮  ○
  ●     │     ○                        ●  │           │  ○
        │                              ╰──╯           ╰──
  (misclassifies edge cases)           (captures non-linear patterns)
```

---

## 4. Side-by-Side Comparison

| | Logistic Regression | MLP (this project) |
|---|---|---|
| Architecture | Input → Sigmoid | Input → 32(ReLU) → 16(ReLU) → Sigmoid |
| Decision boundary | Linear (hyperplane) | Non-linear (learned) |
| Parameters | 7 (6 weights + bias) | ~1,000 |
| Interpretability | High — each weight = feature importance | Low — weights distributed across layers |
| Training speed | Very fast | Fast (small network) |
| Handles feature interaction | ✗ No | ✓ Yes |
| Risk of overfitting | Low | Medium (mitigated by small size here) |
| Typical use case | Baselines, regulated industries | Complex patterns, sufficient data |

---

## 5. Why MLP for Federated Learning?

In the federated setting, each client only sees a subset of the data.
MLP's ability to learn non-linear patterns becomes important because:

- Client 1 (1st class) has a very different survival distribution from Client 3 (3rd class)
- The relationship between features and survival is not purely linear
  (e.g. "female" matters more in 1st class than in 3rd)
- MLP can learn these conditional patterns given enough training signal

LR would converge faster per round, but would hit a lower accuracy ceiling
because it cannot capture cross-feature interactions.

---

## 6. Gradient Flow — How Both Models Learn

Both models are trained by **gradient descent**:

```
For each training step:
  1. Forward pass:   compute prediction  ŷ = model(x)
  2. Compute loss:   L = BinaryCrossEntropy(y, ŷ)
  3. Backward pass:  compute ∂L/∂w for each weight
  4. Update:         w ← w - lr × ∂L/∂w

For LR: one set of weights, one gradient computation.
For MLP: gradients flow backwards through all layers (backpropagation).
```

### Why this matters for Differential Privacy

DP-SGD clips and adds noise to the **gradients** (step 3 above).

- LR has only 7 parameters → 7 gradients → noise is proportionally large
- MLP has ~1,000 parameters → more signal to recover from noise
- In practice, larger models tolerate DP noise better than smaller ones

This is why, even though LR is simpler, MLP is the natural choice
when combining federated learning with differential privacy.

---

## References

- Rumelhart D E, et al. *Learning representations by back-propagating errors.* Nature, 1986.
- Goodfellow I, et al. *Deep Learning.* MIT Press, 2016. (Chapter 6 — MLP)
- Hosmer D W, Lemeshow S. *Applied Logistic Regression.* Wiley, 2000.
- Abadi M, et al. *Deep Learning with Differential Privacy.* ACM CCS, 2016.
