"""
EEG biometric authentication using Riemannian Potato (SSVEP, Unicorn 8-channel).

Pipeline:
  1. Load CSV  (Unicorn export: 'EEG 1' … 'EEG 8', values in nV)
  2. Preprocess: notch 50 Hz + bandpass 5–40 Hz
  3. Epoch into 2-second windows with 50 % overlap
  4. Per epoch: 8×8 covariance (OAS) then TRACE-NORMALIZE
       → removes inter-session amplitude drift, keeps spatial pattern
  5. enroll()  – fit one Potato per person per flickering rate
  6. verify()  – accept if > 50 % of epochs are inliers

Data units: raw Unicorn CSV is in nV. After bandpass the DC is removed;
artifact threshold is therefore in nV too (150 000 nV = 150 µV).
"""

import pickle

import numpy as np
import pandas as pd
from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances
from scipy.signal import butter, filtfilt, iirnotch

# ── configuration ────────────────────────────────────────────────────────────

SFREQ = 250                          # Unicorn sampling rate (Hz)
N_CHANNELS = 8
# CSV columns: 'EEG 1' then ' EEG 2' … ' EEG 8' (leading space from Unicorn export)
# We just take the first N_CHANNELS columns, whatever they're called.

BANDPASS_HZ = (5.0, 40.0)           # covers SSVEP frequencies + harmonics
NOTCH_HZ = 50.0                     # power-line noise
EPOCH_SEC = 2.0
OVERLAP = 0.5                       # 50 % → step = 1 s
Z_THRESHOLD = 2.5                   # z-score cutoff; lower = stricter (FAR/FRR knob)
ARTIFACT_NV = 150_000.0             # 150 µV in nV; epochs exceeding this are dropped
ACCEPT_MAJORITY = 0.5               # fraction of inlier epochs required to accept

# ── data loading ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> np.ndarray:
    """Load Unicorn CSV → (n_channels, n_samples) float64.

    Validates that no channel is flat (dead electrode).
    Unicorn export may have leading spaces in column names — we select by position.
    """
    df = pd.read_csv(path)
    eeg_cols = df.columns[:N_CHANNELS]
    X = df[eeg_cols].to_numpy(dtype=np.float64).T   # (8, n_samples)

    flat = np.where(X.std(axis=1) < 1.0)[0]         # std < 1 nV = dead channel
    if len(flat):
        print(f"WARNING: flat/dead channels at positions {flat.tolist()} in {path}")

    return X


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """Notch + bandpass filter. Must be identical at enrollment and verification."""
    b, a = iirnotch(NOTCH_HZ, Q=30, fs=sfreq)
    X = filtfilt(b, a, X, axis=1)

    lo = BANDPASS_HZ[0] / (sfreq / 2)
    hi = BANDPASS_HZ[1] / (sfreq / 2)
    b, a = butter(4, [lo, hi], btype="band")
    X = filtfilt(b, a, X, axis=1)
    return X


# ── epoching ──────────────────────────────────────────────────────────────────

def epoch(X: np.ndarray, sec: float = EPOCH_SEC, overlap: float = OVERLAP,
          sfreq: float = SFREQ) -> np.ndarray:
    """Continuous signal → (n_epochs, n_channels, n_samples_per_epoch).

    Drops epochs where any channel exceeds ARTIFACT_NV (blinks, muscle noise).
    """
    n = int(sec * sfreq)
    step = max(1, int(n * (1 - overlap)))
    epochs = np.stack([X[:, s:s + n] for s in range(0, X.shape[1] - n + 1, step)])

    keep = np.abs(epochs).max(axis=(1, 2)) < ARTIFACT_NV
    dropped = (~keep).sum()
    if dropped:
        print(f"  artifact rejection: dropped {dropped}/{len(epochs)} epochs")
    return epochs[keep]


# ── covariance ────────────────────────────────────────────────────────────────

def to_covs(epochs: np.ndarray) -> np.ndarray:
    """(n_epochs, 8, n_times) → (n_epochs, 8, 8) trace-normalized SPD matrices.

    OAS shrinkage guarantees well-conditioned matrices for short windows.
    Trace normalization removes global amplitude drift between sessions,
    keeping only the person-specific spatial correlation pattern.
    """
    covs = Covariances(estimator="oas").transform(epochs)
    traces = np.trace(covs, axis1=1, axis2=2)[:, None, None]
    return covs / traces


# ── enrollment ────────────────────────────────────────────────────────────────

def enroll(csv_path: str, z_threshold: float = Z_THRESHOLD) -> Potato:
    """Fit a Potato (personalized fence) from one person's enrollment CSV."""
    X = preprocess(load_csv(csv_path))
    covs = to_covs(epoch(X))
    print(f"  enrollment: {len(covs)} clean epochs from {csv_path}")
    if len(covs) < 15:
        print(f"  WARNING: only {len(covs)} epochs — aim for >= 20-30 for a stable fence")

    potato = Potato(metric="riemann", threshold=z_threshold)
    potato.fit(covs)
    return potato


def enroll_multi(csv_per_rate: dict, z_threshold: float = Z_THRESHOLD) -> dict:
    """Enroll across multiple flickering rates. Returns {rate: Potato}.

    Each rate produces a different spatial SSVEP pattern → separate Potato per rate.
    Requiring all rates to pass lowers FAR significantly.

    Args:
        csv_per_rate: e.g. {'5hz': 'data/raw/enroll_5hz.csv', '10hz': '...', '15hz': '...'}
    """
    return {rate: enroll(path, z_threshold) for rate, path in csv_per_rate.items()}


# ── verification ──────────────────────────────────────────────────────────────

def verify(potato: Potato, csv_path: str,
           accept_majority: float = ACCEPT_MAJORITY) -> dict:
    """Run one login attempt through a single Potato.

    Returns a result dict with decision, accept_rate, mean z-score, epoch count.
    """
    X = preprocess(load_csv(csv_path))
    covs = to_covs(epoch(X))

    labels = potato.predict(covs)       # 1 = inlier (you), 0 = outlier
    zscores = potato.transform(covs)

    accept_rate = float(labels.mean())
    return {
        "decision": "ACCEPT" if accept_rate > accept_majority else "REJECT",
        "accept_rate": round(accept_rate, 3),
        "mean_zscore": round(float(zscores.mean()), 3),
        "n_epochs": len(covs),
    }


def verify_multi(potatoes: dict, csv_per_rate: dict,
                 require: str = "all") -> tuple[str, dict]:
    """Verify across multiple rates.

    Args:
        potatoes:     {rate: Potato}  from enroll_multi()
        csv_per_rate: {rate: csv_path} for the login attempt
        require:      'all' (AND logic, lower FAR) or 'any' (OR logic, lower FRR)

    Returns:
        (overall_decision, {rate: result_dict})
    """
    results = {rate: verify(potatoes[rate], path) for rate, path in csv_per_rate.items()}
    passed = [r["decision"] == "ACCEPT" for r in results.values()]
    ok = all(passed) if require == "all" else any(passed)
    return ("ACCEPT" if ok else "REJECT"), results


# ── enrollment from array (for split-testing) ─────────────────────────────────

def enroll_from_array(X: np.ndarray, z_threshold: float = Z_THRESHOLD) -> Potato:
    """Fit a Potato from an already-loaded (n_channels, n_samples) array."""
    covs = to_covs(epoch(preprocess(X)))
    print(f"  enrollment: {len(covs)} clean epochs")
    potato = Potato(metric="riemann", threshold=z_threshold)
    potato.fit(covs)
    return potato


def verify_from_array(potato: Potato, X: np.ndarray,
                      accept_majority: float = ACCEPT_MAJORITY) -> dict:
    """Verify against an already-loaded (n_channels, n_samples) array."""
    covs = to_covs(epoch(preprocess(X)))
    labels = potato.predict(covs)
    zscores = potato.transform(covs)
    accept_rate = float(labels.mean())
    return {
        "decision": "ACCEPT" if accept_rate > accept_majority else "REJECT",
        "accept_rate": round(accept_rate, 3),
        "mean_zscore": round(float(zscores.mean()), 3),
        "n_epochs": len(covs),
    }


# ── persistence ───────────────────────────────────────────────────────────────

def save_potatoes(potatoes: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(potatoes, f)
    print(f"Saved {list(potatoes.keys())} → {path}")


def load_potatoes(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── example / quick-start ────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── split-test: enroll on first half, verify on held-out second half ──────
    print("=== Split-test: ziyang first-half enroll / second-half verify ===")
    X_full = load_csv("data/raw/1230_ziyang_5hz.csv")
    mid = X_full.shape[1] // 2
    X_enroll, X_verify = X_full[:, :mid], X_full[:, mid:]

    potato_split = enroll_from_array(X_enroll)
    r_self  = verify_from_array(potato_split, X_verify)
    r_chris = verify_from_array(potato_split, load_csv("data/raw/1230_chris_5hz.csv"))
    print(f"  ziyang held-out half : {r_self}")
    print(f"  chris impostor       : {r_chris}")

    # ── full enrollment on all ziyang data ────────────────────────────────────
    print()
    ENROLL_CSVS = {
        "5hz":  "data/raw/1230_ziyang_5hz.csv",
        # "10hz": "data/raw/1230_ziyang_10hz.csv",
        # "15hz": "data/raw/1230_ziyang_15hz.csv",
    }

    print("=== Full enrollment (ziyang) ===")
    potatoes = enroll_multi(ENROLL_CSVS)
    save_potatoes(potatoes, "models/ziyang_potatoes.pkl")

    print("\n=== Self-test on training data (upper bound) ===")
    decision, results = verify_multi(potatoes, ENROLL_CSVS)
    print(f"Decision: {decision}")
    for rate, r in results.items():
        print(f"  {rate}: {r}")

    print("\n=== Impostor: chris (should REJECT) ===")
    IMPOSTOR_CSVS = {
        "5hz": "data/raw/1230_chris_5hz.csv",
    }
    decision, results = verify_multi(potatoes, IMPOSTOR_CSVS)
    print(f"Decision: {decision}")
    for rate, r in results.items():
        print(f"  {rate}: {r}")
