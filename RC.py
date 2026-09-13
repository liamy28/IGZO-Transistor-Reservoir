import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh as geigh
from pathlib import Path

# ============================================================================
# RESERVOIR Comp USING THE MEASURED IGZO FIT PARAMETERS
# ============================================================================
DEVICE = dict(
    I0_rise=4.966e-7,
    A_rise=6.211e-6,
    tau_rise=1.322e-2,
    I0_decay=8.261e-7,
    A1=5.362e-6,
    tau1=3.930e-3,
    A2=3.758e-7,
    tau2=9.092e-2,
)

## ASSUMPTION fractions. possibly contributes to source of inaccuracy
_decay_total = DEVICE['A1'] + DEVICE['A2']
FAST_FRACTION = DEVICE['A1'] / _decay_total
SLOW_FRACTION = DEVICE['A2'] / _decay_total

T, ML, N = 100, 6, 1
MASK = np.random.default_rng(7).uniform(0.1, 1.0, size=ML)
DT_SUB = 0.002

# solution to exponential
def _exact_relaxation(x, target, tau, duration):
    return target + (x - target) * np.exp(-duration / tau)



def reservoir_vector(u):
    u = np.asarray(u, dtype=float)


    u = np.clip(u, 0.0, 1.0)
    drive = (u[:, None] * MASK[None, :]).reshape(-1)
    out = np.empty((T * ML, 1), dtype=float)
    
    tau_rise = DEVICE['tau_rise']
    tau_fast = DEVICE['tau1']
    tau_slow = DEVICE['tau2']
    amplitude = DEVICE['A_rise']
    baseline = DEVICE['I0_decay']

    x_fast = 0.0
    x_slow = 0.0

    for slot, v in enumerate(drive):
        
        # if voltage is applied forever, what would the target result be?
        target_fast = amplitude * FAST_FRACTION * v
        target_slow = amplitude * SLOW_FRACTION * v

        # select tau
        tf = tau_rise if target_fast >= x_fast else tau_fast
        ts = tau_rise if target_slow >= x_slow else tau_slow

        # update x
        x_fast = _exact_relaxation(x_fast, target_fast, tf, DT_SUB)
        x_slow = _exact_relaxation(x_slow, target_slow, ts, DT_SUB)
        out[slot, 0] = baseline + x_fast + x_slow

    
    return out.reshape(-1)


# ============================================================================
# DATA loading
# ============================================================================
DATASET_PATH = Path(__file__).with_name("stars_reservoir_dataset.npz")

with np.load(DATASET_PATH, allow_pickle=False) as dataset:
    curves = dataset["curves"] #(200, 100) - 200 light curves, 100 time points each
    labels = dataset["labels"] # (200,) - class labels (0,1,2,3)
    class_names = dataset["class_names"] # ['I', 'II', 'III', 'IV']
    tr_idx = dataset["train_idx"] # Training indices (45 per class)
    te_idx = dataset["test_idx"] # Test indices (5 per class)


print("Generating reservoir states...")
X_res = np.asarray([reservoir_vector(u) for u in curves]) # (200, 600)
X_raw = np.asarray(curves)
print(f"Reservoir states: shape={X_res.shape}, "
      f"range=[{X_res.min():.3e}, {X_res.max():.3e}] A")


# ============================================================================
# MLP its only 1 hidden layer "the readout" layer
# ============================================================================
def softmax(z):
    """Convert logits to class probabilities."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits, y):
    """Mean multiclass cross-entropy loss."""
    p = softmax(logits)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


class MLP:
    def __init__(self, din, dh=30, dout=4, seed=0):
        # din  = input dimension (600 for reservoir, 100 for raw)
        # dh   = hidden-layer size
        # dout = number of classes
        r = np.random.default_rng(seed)

        # Xavier initialization
        self.W1 = r.uniform(-1, 1, (din, dh)) * np.sqrt(6 / (din + dh))
        self.b1 = np.zeros(dh)
        self.W2 = r.uniform(-1, 1, (dh, dout)) * np.sqrt(6 / (dh + dout))
        self.b2 = np.zeros(dout)

        # Adam optimizer state
        self._adam_t = 0
        self._adam_m = [
            np.zeros_like(p)
            for p in (self.W1, self.b1, self.W2, self.b2)
        ]
        self._adam_v = [
            np.zeros_like(p)
            for p in (self.W1, self.b1, self.W2, self.b2)
        ]

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def step(self, X, y, lr=1e-3, weight_decay=1e-3):
        B = len(y)
        Y = np.eye(4)[y]
        p = softmax(self.forward(X))

        # Backpropagation
        dz2 = (p - Y) / B
        dW2 = self.a1.T @ dz2 + weight_decay * self.W2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1**2)
        dW1 = self.X.T @ dz1 + weight_decay * self.W1
        db1 = dz1.sum(axis=0)

        # Adam update
        self._adam_t += 1
        params = (self.W1, self.b1, self.W2, self.b2)
        grads = (dW1, db1, dW2, db2)

        for i, (param, grad) in enumerate(zip(params, grads)):
            self._adam_m[i] = 0.9 * self._adam_m[i] + 0.1 * grad
            self._adam_v[i] = 0.999 * self._adam_v[i] + 0.001 * grad**2

            m_hat = self._adam_m[i] / (1 - 0.9**self._adam_t)
            v_hat = self._adam_v[i] / (1 - 0.999**self._adam_t)

            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def train_model(
    Xtr,
    ytr,
    Xte,
    yte,
    epochs=300,
    lr=1e-3,
    batch=32,
    seed=0,
    patience=30,
):
    """
    Train using an internal fit/validation split.

    IMPORTANT:
    - The test set is NOT evaluated during training.
    - Early stopping uses validation accuracy only.
    - After training, the weights from the best validation epoch are restored.
    - The held-out test set is then evaluated exactly once.
    """

    net = MLP(Xtr.shape[1], seed=seed)
    r = np.random.default_rng(seed + 1)

    # ------------------------------------------------------------------
    # INTERNAL STRATIFIED FIT / VALIDATION SPLIT
    # ------------------------------------------------------------------
    fit_idx = []
    val_idx = []

    for c in range(4):
        idx_c = np.flatnonzero(ytr == c)
        r.shuffle(idx_c)

        n_val = max(1, int(round(0.15 * len(idx_c))))

        val_idx.extend(idx_c[:n_val])
        fit_idx.extend(idx_c[n_val:])

    fit_idx = np.asarray(fit_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)

    # Histories intentionally contain ONLY fit and validation metrics.
    hist_fit_acc = []
    hist_val_acc = []
    hist_fit_loss = []
    hist_val_loss = []

    best_val = -np.inf
    best_params = None
    best_val_epoch = 0
    best_val_acc = 0.0
    stale = 0

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    for ep in range(epochs):
        idx = r.permutation(fit_idx)

        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            net.step(Xtr[b], ytr[b], lr)

        # Fit metrics
        fit_logits = net.forward(Xtr[fit_idx])
        fit_pred = np.argmax(fit_logits, axis=1)
        fit_acc = np.mean(fit_pred == ytr[fit_idx])
        fit_loss = cross_entropy_loss(fit_logits, ytr[fit_idx])

        # Validation metrics
        val_logits = net.forward(Xtr[val_idx])
        val_pred = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_pred == ytr[val_idx])
        val_loss = cross_entropy_loss(val_logits, ytr[val_idx])

        hist_fit_acc.append(fit_acc)
        hist_val_acc.append(val_acc)
        hist_fit_loss.append(fit_loss)
        hist_val_loss.append(val_loss)

        # Save model whenever validation accuracy reaches a new maximum.
        if val_acc > best_val + 1e-12:
            best_val = val_acc
            best_params = tuple(
                p.copy()
                for p in (net.W1, net.b1, net.W2, net.b2)
            )
            best_val_epoch = ep + 1
            best_val_acc = val_acc
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    # ------------------------------------------------------------------
    # RESTORE THE MODEL FROM THE BEST VALIDATION EPOCH
    # ------------------------------------------------------------------
    net.W1, net.b1, net.W2, net.b2 = best_params

    # Metrics of the restored best-validation model.
    final_fit_acc = np.mean(
        net.predict(Xtr[fit_idx]) == ytr[fit_idx]
    )
    final_val_acc = np.mean(
        net.predict(Xtr[val_idx]) == ytr[val_idx]
    )

    # --------------------------------------------------------------
    # TEST SET IS EVALUATED ONCE, HERE, AFTER MODEL SELECTION.
    # --------------------------------------------------------------
    final_test_acc = np.mean(
        net.predict(Xte) == yte
    )

    print(f"  stopped after {len(hist_fit_acc)} epochs")
    print(
        f"  best validation at epoch {best_val_epoch}: "
        f"{best_val_acc:.3f}"
    )
    print(
        f"  restored model: fit={final_fit_acc:.3f}, "
        f"validation={final_val_acc:.3f}, "
        f"test={final_test_acc:.3f}"
    )

    return {
        "net": net,
        "fit_acc": np.asarray(hist_fit_acc),
        "val_acc": np.asarray(hist_val_acc),
        "fit_loss": np.asarray(hist_fit_loss),
        "val_loss": np.asarray(hist_val_loss),
        "best_val_epoch": best_val_epoch,
        "best_val_acc": best_val_acc,
        "final_fit_acc": final_fit_acc,
        "final_val_acc": final_val_acc,
        "final_test_acc": final_test_acc,
        "fit_idx": fit_idx,
        "val_idx": val_idx,
    }


# ============================================================================
# STANDARDIZATION
# ============================================================================
def standardize_from_train(X, train_indices):
    """Standardize using statistics from the outer training set only."""
    mean = X[train_indices].mean(axis=0)
    scale = X[train_indices].std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (X - mean) / scale


# ============================================================================
# TRAIN BOTH SYSTEMS
# ============================================================================
TRAINING_SEED = 0

print("\n" + "=" * 60)
print("TRAINING WITH RESERVOIR")
print("=" * 60)

X_res_scaled = standardize_from_train(X_res, tr_idx)

res = train_model(
    X_res_scaled[tr_idx],
    labels[tr_idx],
    X_res_scaled[te_idx],
    labels[te_idx],
    seed=TRAINING_SEED,
)

print("\n" + "=" * 60)
print("TRAINING WITHOUT RESERVOIR (RAW CURVES)")
print("=" * 60)

X_raw_scaled = standardize_from_train(X_raw, tr_idx)

raw = train_model(
    X_raw_scaled[tr_idx],
    labels[tr_idx],
    X_raw_scaled[te_idx],
    labels[te_idx],
    seed=TRAINING_SEED,
)

net_res = res["net"]
net_raw = raw["net"]


# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"With Reservoir    - "
    f"Fit: {res['final_fit_acc']:.3f}, "
    f"Validation: {res['final_val_acc']:.3f}, "
    f"Test: {res['final_test_acc']:.3f}"
)

print(
    f"Without Reservoir - "
    f"Fit: {raw['final_fit_acc']:.3f}, "
    f"Validation: {raw['final_val_acc']:.3f}, "
    f"Test: {raw['final_test_acc']:.3f}"
)

print(
    f"Test improvement: "
    f"{(res['final_test_acc'] - raw['final_test_acc']) * 100:.1f} "
    f"percentage points"
)

print(
    f"\nBest validation (Reservoir): "
    f"epoch {res['best_val_epoch']}, "
    f"acc={res['best_val_acc']:.3f}"
)

print(
    f"Best validation (No reservoir): "
    f"epoch {raw['best_val_epoch']}, "
    f"acc={raw['best_val_acc']:.3f}"
)


# ============================================================================
# LDA
# ============================================================================
def lda(Xtr, ytr, Xall, n=2, reg=1e-3):
    mu = Xtr.mean(axis=0)
    Sw = np.zeros((Xtr.shape[1],) * 2)
    Sb = np.zeros_like(Sw)

    for c in range(4):
        Xc = Xtr[ytr == c]
        d = Xc - Xc.mean(axis=0)
        Sw += d.T @ d
        Sb += len(Xc) * np.outer(
            Xc.mean(axis=0) - mu,
            Xc.mean(axis=0) - mu,
        )

    Sw += reg * np.eye(Sw.shape[0])

    _, V = geigh(Sb, Sw)
    W = V[:, -n:][:, ::-1]

    return (Xall - mu) @ W


Z = lda(
    X_res_scaled[tr_idx],
    labels[tr_idx],
    X_res_scaled,
)

# Confusion matrix is based on the FINAL restored best-validation model.
pred_te = net_res.predict(
    X_res_scaled[te_idx]
)

cm = np.zeros((4, 4))

for true_label, pred_label in zip(
    labels[te_idx],
    pred_te,
):
    cm[true_label, pred_label] += 1

cmn = cm / cm.sum(axis=1, keepdims=True)


# ============================================================================
# FIGURES
#
# Test-set performance is deliberately NOT plotted across epochs.
# The test set is only evaluated after model selection.
# ============================================================================
fig, ax = plt.subplots(2, 2, figsize=(14, 10))


# ---------------------------------------------------------------------------
# LDA scatter of reservoir encodings
# ---------------------------------------------------------------------------
colors = ["blue", "red", "green", "orange"]

for c in range(4):
    mask = labels == c

    ax[0, 0].scatter(
        Z[mask, 0],
        Z[mask, 1],
        s=10,
        alpha=0.6,
        color=colors[c],
        label=class_names[c],
    )

ax[0, 0].set(
    xlabel="LD1",
    ylabel="LD2",
    title="(a) LDA of reservoir encodings",
)

ax[0, 0].legend(fontsize=7)
ax[0, 0].grid(alpha=0.3)


# ---------------------------------------------------------------------------
# FIT + VALIDATION ACCURACY VS EPOCH
# ---------------------------------------------------------------------------
epochs_res = np.arange(1, len(res["fit_acc"]) + 1)
epochs_raw = np.arange(1, len(raw["fit_acc"]) + 1)

ax[0, 1].plot(
    epochs_res,
    res["fit_acc"],
    linestyle=":",
    label="Fit (reservoir)",
)

ax[0, 1].plot(
    epochs_res,
    res["val_acc"],
    linestyle="-",
    label="Validation (reservoir)",
)

ax[0, 1].plot(
    epochs_raw,
    raw["fit_acc"],
    linestyle=":",
    label="Fit (no reservoir)",
)

ax[0, 1].plot(
    epochs_raw,
    raw["val_acc"],
    linestyle="-",
    label="Validation (no reservoir)",
)

ax[0, 1].axvline(
    res["best_val_epoch"],
    linestyle="--",
    alpha=0.6,
    label=f"Best res val: epoch {res['best_val_epoch']}",
)

ax[0, 1].axvline(
    raw["best_val_epoch"],
    linestyle="--",
    alpha=0.6,
    label=f"Best raw val: epoch {raw['best_val_epoch']}",
)

ax[0, 1].set(
    xlabel="Epoch",
    ylabel="Accuracy",
    ylim=(0.2, 1.01),
    title="(b) Fit and validation accuracy",
)

ax[0, 1].legend(fontsize=7)
ax[0, 1].grid(alpha=0.3)


# ---------------------------------------------------------------------------
# 3. FIT + VALIDATION CROSS-ENTROPY LOSS VS EPOCH
# ---------------------------------------------------------------------------
ax[1, 0].plot(
    epochs_res,
    res["fit_loss"],
    linestyle=":",
    label="Fit loss (reservoir)",
)

ax[1, 0].plot(
    epochs_res,
    res["val_loss"],
    linestyle="-",
    label="Validation loss (reservoir)",
)

ax[1, 0].plot(
    epochs_raw,
    raw["fit_loss"],
    linestyle=":",
    label="Fit loss (no reservoir)",
)

ax[1, 0].plot(
    epochs_raw,
    raw["val_loss"],
    linestyle="-",
    label="Validation loss (no reservoir)",
)

ax[1, 0].axvline(
    res["best_val_epoch"],
    linestyle="--",
    alpha=0.6,
)

ax[1, 0].axvline(
    raw["best_val_epoch"],
    linestyle="--",
    alpha=0.6,
)

ax[1, 0].set(
    xlabel="Epoch",
    ylabel="Cross-entropy loss",
    title="(c) Fit and validation loss",
)

ax[1, 0].legend(fontsize=7)
ax[1, 0].grid(alpha=0.3)


# ---------------------------------------------------------------------------
# FINAL TEST CONFUSION MATRIX
# ---------------------------------------------------------------------------
im = ax[1, 1].imshow(
    cmn,
    cmap="Blues",
    vmin=0,
    vmax=1,
)

ax[1, 1].set(
    xlabel="Predicted",
    ylabel="True",
    title=(
        "(d) Reservoir confusion matrix "
        f"(final test={res['final_test_acc']:.3f})"
    ),
)

ax[1, 1].set_xticks(range(4))
ax[1, 1].set_yticks(range(4))
ax[1, 1].set_xticklabels(class_names)
ax[1, 1].set_yticklabels(class_names)

for i in range(4):
    for j in range(4):
        ax[1, 1].text(
            j,
            i,
            f"{cmn[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if cmn[i, j] > 0.5 else "black",
        )

fig.colorbar(im, ax=ax[1, 1])

plt.tight_layout()

plt.savefig(
    "reservoir_results_clean_evaluation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()



# ============================================================================
# ADDITIONAL FIGURE: RESERVOIR STATES
# ============================================================================
# Plot one representative light curve from each class and the corresponding
# six virtual-node reservoir states.
#
# X_res has shape: (number_of_light_curves, T*ML)
# For each star, reshape the 600 reservoir outputs into (T, ML), where each
# row corresponds to one original phase point and each column is one masked
# virtual node.

fig_states, axes_states = plt.subplots(
    len(class_names), 2,
    figsize=(13, 3.2 * len(class_names)),
    sharex=False
)

phase_axis = np.arange(T) / T

for c in range(len(class_names)):
    # Choose the first test example from this class if available.
    class_test = te_idx[labels[te_idx] == c]
    if len(class_test) > 0:
        sample_idx = class_test[0]
    else:
        sample_idx = np.flatnonzero(labels == c)[0]

    # ----- Left: original phase-folded light curve -----
    axes_states[c, 0].plot(phase_axis, curves[sample_idx], linewidth=1.5)
    axes_states[c, 0].set_ylabel(str(class_names[c]))
    axes_states[c, 0].grid(alpha=0.3)

    if c == 0:
        axes_states[c, 0].set_title("Input phase-folded light curve")
    if c == len(class_names) - 1:
        axes_states[c, 0].set_xlabel("Phase")

    # ----- Right: reservoir states -----
    # reservoir_vector() stores values in the order:
    # time point 0: node 0...ML-1,
    # time point 1: node 0...ML-1, etc.
    states = X_res[sample_idx].reshape(T, ML)

    for node in range(ML):
        axes_states[c, 1].plot(
            phase_axis,
            states[:, node] * 1e6,   # convert A -> microampere
            linewidth=1.0,
            alpha=0.8,
            label=f"Node {node + 1}" if c == 0 else None
        )

    axes_states[c, 1].grid(alpha=0.3)

    if c == 0:
        axes_states[c, 1].set_title("IGZO reservoir states")
        axes_states[c, 1].legend(
            ncol=3,
            fontsize=8,
            loc="best"
        )

    if c == len(class_names) - 1:
        axes_states[c, 1].set_xlabel("Input phase point")

    axes_states[c, 1].set_ylabel("Current (µA)")

fig_states.suptitle(
    "Representative input light curves and reservoir states",
    fontsize=14,
    y=1.01
)

plt.tight_layout()
plt.savefig(
    "reservoir_states_by_class.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

