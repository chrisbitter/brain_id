"""
EEG biometric authentication using Riemannian Potato (SSVEP, Unicorn 8-channel).

Pipeline:
  1. Load CSV        (Unicorn export: 'EEG 1' … 'EEG 8', values in nV)
  2. Channel select  (keep occipital/parietal: Pz, PO7, Oz, PO8 = ch5–ch8)
  3. Preprocess      notch 50 Hz → narrow bandpass ±2 Hz around SSVEP freq → CAR
  4. Downsample      250 Hz → 100 Hz (sufficient for 5–15 Hz SSVEP; better cov. conditioning)
  5. Epoch           2-second windows, 50 % overlap, artifact rejection
  6. Covariance      OAS shrinkage → trace-normalise
  7. enroll()        fit Potato per person per flickering rate
  8. verify()        accept if > 50 % of epochs are inliers

Reference for preprocessing choices
-------------------------------------
Stieger et al., "Biosensors" 2021, 11, 404 — motor imagery on Unicorn Hybrid Black:
  - 4th-order Butterworth bandpass 1–30 Hz on same device/electrode layout
  - Downsample to 100 Hz post-filter
  - Electrode map (Fig. 2): Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 (confirms our OCC_CHANNELS)
We adopt 1 Hz lower cutoff and 100 Hz resampling for the broadband fallback.
Narrow SSVEP bandpass (±2 Hz around stimulus) supersedes the broadband filter when
a stimulus frequency is known.

Data units: raw Unicorn CSV is in nV. Artifact threshold is therefore in nV too.

Preprocessing improvements over broadband baseline:
  - Narrow bandpass: keeps only the SSVEP-driven spatial response at each rate,
    removing broadband session drift that dominates PCA
  - CAR (Common Average Reference): subtracts cross-channel mean at each sample,
    removes global amplitude shifts between sessions
  - Channel selection: drops frontal/motor channels (ch1–ch4) that carry no
    SSVEP signal but add session-variable noise
"""

import pickle

import numpy as np
import pandas as pd
from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances
from scipy.signal import butter, detrend, filtfilt, iirnotch, resample

# ── configuration ────────────────────────────────────────────────────────────

SFREQ        = 250            # Unicorn acquisition rate (Hz)
SFREQ_DS     = 100            # downsample target (Hz); paper: 250→100 on same device
                              # sufficient for 5–15 Hz SSVEP (Nyquist=50 Hz);
                              # smaller epochs (200 vs 500 samples) → better OAS conditioning
N_CHANNELS   = 8
NOTCH_HZ     = 50.0           # power-line noise
NARROW_BW    = 2.0            # ± Hz around SSVEP frequency for narrow bandpass
BANDPASS_HZ  = (1.0, 30.0)   # broadband fallback — paper uses 1–30 Hz on Unicorn
EPOCH_SEC    = 2.0
OVERLAP      = 0.5            # 50 % → step = 1 s
Z_THRESHOLD  = 2.5
ARTIFACT_NV  = 150_000.0     # 150 µV in nV
ACCEPT_MAJORITY = 0.5

# Occipital/parietal channels (0-indexed): Pz(4), PO7(5), Oz(6), PO8(7)
# These are the channels that carry the SSVEP visual response.
# Frontal/motor channels ch1–ch4 (indices 0–3) are dropped.
OCC_CHANNELS = [4, 5, 6, 7]

# Maps rate key → SSVEP stimulus frequency in Hz
RATE_TO_FREQ = {"5hz": 5.0, "10hz": 10.0, "15hz": 15.0}

# ── data loading ──────────────────────────────────────────────────────────────

def load_csv(path: str, channels: list = None) -> np.ndarray:
    """Load Unicorn CSV → (n_channels, n_samples) float64.

    Args:
        channels: list of 0-based channel indices to keep (default: all 8).
                  Use OCC_CHANNELS to keep only occipital/parietal channels.
    """
    df = pd.read_csv(path)
    eeg_cols = df.columns[:N_CHANNELS]
    X = df[eeg_cols].to_numpy(dtype=np.float64).T   # (8, n_samples)

    flat = np.where(X.std(axis=1) < 1.0)[0]
    if len(flat):
        print(f"WARNING: flat/dead channels {flat.tolist()} in {path}")

    if channels is not None:
        X = X[channels]

    return X


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X: np.ndarray, ssvep_freq: float = None,
               sfreq: float = SFREQ, target_sfreq: float = SFREQ_DS) -> np.ndarray:
    """Notch → bandpass → downsample → CAR.

    Follows Biosensors 2021, 11, 404 (same Unicorn device):
      bandpass 1–30 Hz, then 250→100 Hz downsample.
    When ssvep_freq is given, narrow ±2 Hz bandpass replaces broadband.
    """
    # 1. Notch 50 Hz
    b, a = iirnotch(NOTCH_HZ, Q=30, fs=sfreq)
    X = filtfilt(b, a, X, axis=1)

    # 2. Bandpass (doubles as anti-alias filter before downsampling)
    if ssvep_freq is not None:
        lo = max(0.5, ssvep_freq - NARROW_BW) / (sfreq / 2)
        hi = min(sfreq / 2 - 1, ssvep_freq + NARROW_BW) / (sfreq / 2)
    else:
        lo = BANDPASS_HZ[0] / (sfreq / 2)
        hi = BANDPASS_HZ[1] / (sfreq / 2)
    b, a = butter(4, [lo, hi], btype="band")
    X = filtfilt(b, a, X, axis=1)

    # 3. Downsample 250 → 100 Hz
    if target_sfreq != sfreq:
        n_out = int(X.shape[1] * target_sfreq / sfreq)
        X = resample(X, n_out, axis=1)

    # 4. CAR
    X = X - X.mean(axis=0, keepdims=True)
    return X


# ── epoching ──────────────────────────────────────────────────────────────────

def epoch(X: np.ndarray, sec: float = EPOCH_SEC, overlap: float = OVERLAP,
          sfreq: float = SFREQ_DS) -> np.ndarray:
    """Continuous (n_channels, n_samples) → (n_epochs, n_channels, n_times).

    Detrends each epoch (removes linear drift within window) and rejects
    epochs where any channel exceeds ARTIFACT_NV.
    """
    n = int(sec * sfreq)
    step = max(1, int(n * (1 - overlap)))
    epochs = np.stack([X[:, s:s + n] for s in range(0, X.shape[1] - n + 1, step)])

    # detrend each epoch along time axis
    epochs = detrend(epochs, axis=2)

    keep = np.abs(epochs).max(axis=(1, 2)) < ARTIFACT_NV
    dropped = (~keep).sum()
    if dropped:
        print(f"  artifact rejection: dropped {dropped}/{len(epochs)} epochs")
    return epochs[keep]


# ── covariance ────────────────────────────────────────────────────────────────

def to_covs(epochs: np.ndarray) -> np.ndarray:
    """(n_epochs, n_ch, n_times) → (n_epochs, n_ch, n_ch) trace-normalised SPD.

    OAS shrinkage guarantees well-conditioned matrices for short windows.
    Trace normalization removes remaining amplitude drift between sessions.
    """
    covs = Covariances(estimator="oas").transform(epochs)
    traces = np.trace(covs, axis1=1, axis2=2)[:, None, None]
    return covs / traces


# ── internal pipeline helper ──────────────────────────────────────────────────

def _pipeline(csv_path: str, ssvep_freq: float = None,
              channels: list = OCC_CHANNELS) -> np.ndarray:
    """Full preprocessing pipeline: CSV → covariance matrices."""
    return to_covs(epoch(preprocess(load_csv(csv_path, channels), ssvep_freq)))


# ── enrollment ────────────────────────────────────────────────────────────────

def enroll(csv_path: str, ssvep_freq: float = None,
           channels: list = OCC_CHANNELS,
           z_threshold: float = Z_THRESHOLD) -> Potato:
    """Fit a Potato from one person's enrollment CSV."""
    covs = _pipeline(csv_path, ssvep_freq, channels)
    print(f"  enrollment: {len(covs)} clean epochs  [{csv_path.split('/')[-1]}]")
    if len(covs) < 15:
        print(f"  WARNING: only {len(covs)} epochs — aim for >= 20-30")
    potato = Potato(metric="riemann", threshold=z_threshold)
    potato.fit(covs)
    return potato


def enroll_multi(csv_per_rate: dict, channels: list = OCC_CHANNELS,
                 z_threshold: float = Z_THRESHOLD) -> dict:
    """Enroll across multiple rates. Automatically uses narrow bandpass per rate.

    Args:
        csv_per_rate: {'5hz': path, '10hz': path, '15hz': path}
    Returns:
        {rate: Potato}
    """
    return {
        rate: enroll(path, RATE_TO_FREQ.get(rate), channels, z_threshold)
        for rate, path in csv_per_rate.items()
    }


# ── verification ──────────────────────────────────────────────────────────────

def verify(potato: Potato, csv_path: str, ssvep_freq: float = None,
           channels: list = OCC_CHANNELS,
           accept_majority: float = ACCEPT_MAJORITY) -> dict:
    """Run one login attempt through a single Potato."""
    covs = _pipeline(csv_path, ssvep_freq, channels)
    labels  = potato.predict(covs)
    zscores = potato.transform(covs)
    accept_rate = float(labels.mean())
    return {
        "decision":    "ACCEPT" if accept_rate > accept_majority else "REJECT",
        "accept_rate": round(accept_rate, 3),
        "mean_zscore": round(float(zscores.mean()), 3),
        "n_epochs":    len(covs),
    }


def verify_multi(potatoes: dict, csv_per_rate: dict,
                 channels: list = OCC_CHANNELS,
                 require: str = "all") -> tuple[str, dict]:
    """Verify across multiple rates (AND/OR logic).

    Args:
        require: 'all' → AND (lower FAR), 'any' → OR (lower FRR)
    """
    results = {
        rate: verify(potatoes[rate], path, RATE_TO_FREQ.get(rate), channels)
        for rate, path in csv_per_rate.items()
    }
    passed = [r["decision"] == "ACCEPT" for r in results.values()]
    ok = all(passed) if require == "all" else any(passed)
    return ("ACCEPT" if ok else "REJECT"), results


# ── enrollment / verification from pre-loaded arrays ─────────────────────────

def enroll_from_array(X: np.ndarray, ssvep_freq: float = None,
                      z_threshold: float = Z_THRESHOLD) -> Potato:
    """Fit a Potato from an already-loaded (n_channels, n_samples) array.
    Channel selection must be applied before calling this.
    """
    covs = to_covs(epoch(preprocess(X, ssvep_freq)))
    print(f"  enrollment: {len(covs)} clean epochs")
    potato = Potato(metric="riemann", threshold=z_threshold)
    potato.fit(covs)
    return potato


def verify_from_array(potato: Potato, X: np.ndarray, ssvep_freq: float = None,
                      accept_majority: float = ACCEPT_MAJORITY) -> dict:
    """Verify against an already-loaded (n_channels, n_samples) array."""
    covs = to_covs(epoch(preprocess(X, ssvep_freq)))
    labels  = potato.predict(covs)
    zscores = potato.transform(covs)
    accept_rate = float(labels.mean())
    return {
        "decision":    "ACCEPT" if accept_rate > accept_majority else "REJECT",
        "accept_rate": round(accept_rate, 3),
        "mean_zscore": round(float(zscores.mean()), 3),
        "n_epochs":    len(covs),
    }


# ── persistence ───────────────────────────────────────────────────────────────

def save_potatoes(potatoes: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(potatoes, f)
    print(f"Saved {list(potatoes.keys())} → {path}")


def load_potatoes(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ENROLL = {
        "10hz": "data/raw/1230_ziyang_10hz.csv",
        "15hz": "data/raw/1230_ziyang_15hz.csv",
    }
    IMPOSTOR = {
        "10hz": "data/raw/1400_chris_10hz.csv",
        "15hz": "data/raw/1400_chris_15hz.csv",
    }
    SELF_1400 = {
        "10hz": "data/raw/1400_ziyang_10hz.csv",
        "15hz": "data/raw/1400_ziyang_15hz.csv",
    }

    print("=== Enrolling ziyang 1230 (narrow bandpass + CAR + occipital channels) ===")
    potatoes = enroll_multi(ENROLL)
    save_potatoes(potatoes, "models/ziyang_potatoes_v2.pkl")

    print("\n=== ziyang 1230 self-test (should ACCEPT) ===")
    d, r = verify_multi(potatoes, ENROLL)
    print(f"  {d}"); [print(f"  {k}: {v}") for k, v in r.items()]

    print("\n=== ziyang 1400 cross-session (should ACCEPT) ===")
    d, r = verify_multi(potatoes, SELF_1400)
    print(f"  {d}"); [print(f"  {k}: {v}") for k, v in r.items()]

    print("\n=== chris 1400 impostor (should REJECT) ===")
    d, r = verify_multi(potatoes, IMPOSTOR)
    print(f"  {d}"); [print(f"  {k}: {v}") for k, v in r.items()]
