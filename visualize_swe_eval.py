"""
SWE rollout evaluation visualizer — run cell-by-cell in Jupyter.

Artifacts produced by evaluate.py (time_variate=true):
  traj.pt          — {"pred": [N, T, 1, H, W], "true": [N, T-1, 1, H, W]}
                     pred[:,0]  = initial state x_0 (ground truth)
                     pred[:,1:] = model predictions at each rollout step
  eval_metrics.json — {"rel_l2": [...T-1 floats], "mse": [...], "l_inf": [...]}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
EVAL_DIR   = Path("evals/swe-unet-YYYY-MM-DD_HH-MM-SS")   # <-- set this
SAMPLE_IDX = 0        # which trajectory to plot
SNAP_STEPS = 5        # number of evenly-spaced time steps to show in the grid
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Load ───────────────────────────────────────────────────────────────────
data    = torch.load(EVAL_DIR / "traj.pt", map_location="cpu")
metrics = json.loads((EVAL_DIR / "eval_metrics.json").read_text())

pred = data["pred"]   # [N, T,   1, H, W]
true = data["true"]   # [N, T-1, 1, H, W]

N, T, _, H, W = pred.shape
T_steps = T - 1      # number of predicted steps (pred[:,0] is x_0)

print(f"Trajectories : {N}")
print(f"Steps (T-1)  : {T_steps}")
print(f"Spatial      : {H} × {W}")
print(f"Metrics keys : {list(metrics.keys())}")


# ── 2. Per-step metric curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
if len(metrics) == 1:
    axes = [axes]

steps = np.arange(T_steps)
for ax, (name, values) in zip(axes, metrics.items()):
    ax.plot(steps, values, linewidth=1.5)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-step rollout metrics", fontsize=13)
fig.tight_layout()
plt.show()


# ── 3. Trajectory snapshots for one sample ────────────────────────────────────
snap_indices = np.linspace(0, T_steps - 1, SNAP_STEPS, dtype=int)

# pred[:,1+t] corresponds to true[:,t]
p = pred[SAMPLE_IDX, 1:].squeeze(1).numpy()   # [T-1, H, W]
t = true[SAMPLE_IDX].squeeze(1).numpy()        # [T-1, H, W]
err = np.abs(p - t)                            # [T-1, H, W]

vmin = min(p.min(), t.min())
vmax = max(p.max(), t.max())
emax = err.max()

fig, axes = plt.subplots(3, SNAP_STEPS, figsize=(3 * SNAP_STEPS, 8))
row_labels = ["Ground truth", "Prediction", "Abs. error"]

for col, step in enumerate(snap_indices):
    for row, (field, vlo, vhi, cmap) in enumerate([
        (t[step],   vmin, vmax, "viridis"),
        (p[step],   vmin, vmax, "viridis"),
        (err[step], 0,    emax, "hot"),
    ]):
        im = axes[row, col].imshow(field, vmin=vlo, vmax=vhi, cmap=cmap, origin="lower")
        axes[row, col].set_title(f"step {step}" if row == 0 else "")
        axes[row, col].axis("off")
        fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=11)

fig.suptitle(f"Sample {SAMPLE_IDX} — ground truth / prediction / error", fontsize=13)
fig.tight_layout()
plt.show()


# ── 4. Error growth over time ─────────────────────────────────────────────────
# Per-sample mean absolute error at each step, for a handful of trajectories
n_show = min(5, N)
step_mae = np.abs(
    pred[: n_show, 1:].squeeze(2).numpy() - true[:n_show].squeeze(2).numpy()
).mean(axis=(-1, -2))   # [n_show, T-1]

fig, ax = plt.subplots(figsize=(7, 4))
for i in range(n_show):
    ax.plot(steps, step_mae[i], alpha=0.6, label=f"sample {i}")
ax.plot(steps, step_mae.mean(axis=0), color="black", linewidth=2.5, label="mean")
ax.set_xlabel("Rollout step")
ax.set_ylabel("Mean absolute error")
ax.set_title("Error growth over rollout")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
