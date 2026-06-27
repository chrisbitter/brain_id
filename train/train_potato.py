"""
Train a Riemannian Potato one-class classifier per SSVEP flickering rate.

Pipeline per rate:
  raw CSV (n_samples, 8)
    → sliding windows (n_windows, 8, 500)
    → Covariances (Ledoit-Wolf)  →  (n_windows, 8, 8) SPD matrices
    → Potato (one-class, z-score threshold)

At login, feed new EEG windows through all three pipelines.
All three must predict +1 (inlier) to grant access.
"""

import os
import pickle

import numpy as np
import pandas as pd
from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline

SFREQ = 250
WINDOW_SEC = 2.0
OVERLAP_SEC = 1.0  # → step = 1 sec

CH_COLS = [f"ch{i}" for i in range(1, 9)]

RATES = {
    "5hz": "data/raw/1230_ziyang_5hz.csv",
    "10hz": "data/raw/1230_ziyang_10hz.csv",
    "15hz": "data/raw/1230_ziyang_15hz.csv",
}


def load_csv(path: str) -> np.ndarray:
    """Return raw EEG as (n_samples, n_channels) float64."""
    df = pd.read_csv(path)
    return df[CH_COLS].to_numpy(dtype=np.float64)


def make_windows(X: np.ndarray, window_samples: int, step_samples: int) -> np.ndarray:
    """Slide a window over (n_samples, n_channels) → (n_windows, n_channels, window_samples)."""
    windows = [
        X[start : start + window_samples].T
        for start in range(0, len(X) - window_samples + 1, step_samples)
    ]
    return np.array(windows)


def build_pipeline(threshold: float = 2.5):
    """
    Covariances(lwf): (n_windows, 8, 500) → (n_windows, 8, 8) SPD matrices
    Potato:           (n_windows, 8, 8)   → predict +1 / -1
    """
    return make_pipeline(
        Covariances(estimator="lwf"),
        Potato(metric="riemann", threshold=threshold),
    )


def main():
    window_samples = int(SFREQ * WINDOW_SEC)     # 500 samples
    step_samples = int(SFREQ * (WINDOW_SEC - OVERLAP_SEC))  # 250 samples

    os.makedirs("models", exist_ok=True)
    pipelines = {}

    for rate, csv_path in RATES.items():
        print(f"\n[{rate}] loading {csv_path}")
        X = load_csv(csv_path)
        X_windows = make_windows(X, window_samples, step_samples)
        print(f"  windows: {X_windows.shape}  (n_windows, n_channels, n_times)")

        pipe = build_pipeline(threshold=2.5)
        pipe.fit(X_windows)
        pipelines[rate] = pipe

        # report z-score distribution on training data
        potato: Potato = pipe.named_steps["potato"]
        z_scores = potato.transform(pipe.named_steps["covariances"].transform(X_windows))
        print(f"  training z-scores — mean: {z_scores.mean():.3f}, std: {z_scores.std():.3f}, max: {z_scores.max():.3f}")
        inlier_rate = (pipe.predict(X_windows) == 1).mean()
        print(f"  inlier rate on training data: {inlier_rate:.1%}")

    model_path = "models/potato_pipelines.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipelines, f)
    print(f"\nSaved all three pipelines → {model_path}")


if __name__ == "__main__":
    main()
