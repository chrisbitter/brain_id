"""
Visualize ziyang vs chris EEG data at 5 Hz SSVEP.

Produces 4 plots:
  1. PSD overlay per channel  – confirms SSVEP peak presence
  2. Correlation matrix heatmaps  – the spatial fingerprint the Potato uses
  3. Z-score distributions  – Potato separation between the two persons
  4. Raw signal snippet  – sanity check on signal quality
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch

from auth import load_csv, preprocess, to_covs, epoch, load_potatoes

# ── data ─────────────────────────────────────────────────────────────────────

SFREQ = 250
CH_NAMES = ["EEG 1", "EEG 2", "EEG 3", "EEG 4", "EEG 5", "EEG 6", "EEG 7", "EEG 8"]
COLORS = {"ziyang": "#2196F3", "chris": "#F44336"}

raw = {
    "ziyang": load_csv("data/raw/1230_ziyang_5hz.csv"),
    "chris":  load_csv("data/raw/1230_chris_5hz.csv"),
}
proc = {name: preprocess(X) for name, X in raw.items()}

potatoes = load_potatoes("models/ziyang_potatoes.pkl")
potato = potatoes["5hz"]

# ── 1. PSD per channel ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True, sharex=True)
fig.suptitle("PSD per channel — ziyang vs chris (5 Hz SSVEP)", fontsize=13)

for ch_idx, ax in enumerate(axes.flat):
    for name, X in proc.items():
        freqs, psd = welch(X[ch_idx], fs=SFREQ, nperseg=SFREQ * 2)
        mask = freqs <= 50
        ax.semilogy(freqs[mask], psd[mask], color=COLORS[name],
                    alpha=0.85, linewidth=1.2, label=name)

    # mark SSVEP fundamental + harmonics
    for h in [5, 10, 15, 20, 25]:
        ax.axvline(h, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    ax.set_title(f"ch {ch_idx + 1}", fontsize=9)
    ax.set_xlabel("Hz", fontsize=8)
    if ch_idx % 4 == 0:
        ax.set_ylabel("PSD", fontsize=8)

axes.flat[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("reports/figures/01_psd_comparison.png", dpi=150)
print("Saved 01_psd_comparison.png")

# ── 2. Correlation matrix heatmaps ───────────────────────────────────────────
# Use the Riemannian mean covariance for ziyang; use the mean epoch cov for chris

def mean_corr(X):
    """Mean trace-normalised covariance → correlation matrix (easier to read)."""
    covs = to_covs(epoch(X))
    C = covs.mean(axis=0)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Mean channel correlation matrix — 5 Hz SSVEP", fontsize=13)

corr = {name: mean_corr(X) for name, X in proc.items()}

for ax, (name, C) in zip(axes[:2], corr.items()):
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(name, fontsize=11)
    ax.set_xticks(range(8)); ax.set_xticklabels([f"ch{i+1}" for i in range(8)], fontsize=7)
    ax.set_yticks(range(8)); ax.set_yticklabels([f"ch{i+1}" for i in range(8)], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)

diff = corr["ziyang"] - corr["chris"]
im = axes[2].imshow(diff, vmin=-1, vmax=1, cmap="RdBu_r")
axes[2].set_title("difference  (ziyang − chris)", fontsize=11)
axes[2].set_xticks(range(8)); axes[2].set_xticklabels([f"ch{i+1}" for i in range(8)], fontsize=7)
axes[2].set_yticks(range(8)); axes[2].set_yticklabels([f"ch{i+1}" for i in range(8)], fontsize=7)
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig("reports/figures/02_correlation_matrices.png", dpi=150)
print("Saved 02_correlation_matrices.png")

# ── 3. Z-score distributions ──────────────────────────────────────────────────

z_scores = {}
for name, X in proc.items():
    covs = to_covs(epoch(X))
    z_scores[name] = potato.transform(covs)

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("Riemannian Potato z-score distribution — 5 Hz SSVEP", fontsize=13)

bins = np.linspace(-3, 8, 40)
for name, z in z_scores.items():
    ax.hist(z, bins=bins, alpha=0.6, color=COLORS[name], label=f"{name}  (n={len(z)})",
            edgecolor="white", linewidth=0.4)

ax.axvline(2.5, color="black", linewidth=1.5, linestyle="--", label="threshold 2.5")
ax.set_xlabel("z-score (Riemannian distance from ziyang mean)", fontsize=10)
ax.set_ylabel("epoch count", fontsize=10)
ax.legend(fontsize=9)
ax.text(2.6, ax.get_ylim()[1] * 0.92, "REJECT →", fontsize=8, color="gray")
ax.text(2.4, ax.get_ylim()[1] * 0.82, "← ACCEPT", fontsize=8, color="gray", ha="right")

plt.tight_layout()
plt.savefig("reports/figures/03_zscore_distribution.png", dpi=150)
print("Saved 03_zscore_distribution.png")

# ── 4. Raw signal snippet (2 seconds) ────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=False)
fig.suptitle("Preprocessed signal — 2-second snippet (5 Hz SSVEP)", fontsize=13)

SNIPPET = SFREQ * 2  # 2 seconds
SKIP = SFREQ * 2    # skip first 2 sec to avoid filter transient

for ax, (name, X) in zip(axes, proc.items()):
    snippet = X[:, SKIP:SKIP + SNIPPET]
    t = np.arange(SNIPPET) / SFREQ

    # stack channels with offset for readability
    spacing = np.abs(snippet).max() * 1.5
    for ch_idx in range(8):
        ax.plot(t, snippet[ch_idx] + ch_idx * spacing,
                color=COLORS[name], linewidth=0.7, alpha=0.85)
        ax.text(-0.05, ch_idx * spacing, f"ch{ch_idx+1}", fontsize=7,
                ha="right", va="center", transform=ax.get_yaxis_transform())

    ax.set_title(name, fontsize=10)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, 2)

plt.tight_layout()
plt.savefig("reports/figures/04_signal_snippet.png", dpi=150)
print("Saved 04_signal_snippet.png")

plt.show()
