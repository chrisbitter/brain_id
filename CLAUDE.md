# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Use EEG signal uniqueness as a biometric authentication method. Each person's EEG response to SSVEP (Steady-State Visual Evoked Potentials) stimuli at different frequencies is unique enough to serve as a login credential — exploiting inter-individual variability as a feature rather than a bug.

## EEG Device: g.tec Unicorn Hybrid Black

- **Channels:** 8 EEG + 3 accelerometer + 3 gyroscope channels
- **Signal quality:** 24-bit resolution, 250 Hz sampling rate
- **Electrodes:** Hybrid (dry or wet with conductive gel)
- **Connectivity:** Bluetooth
- **APIs:** Python, C, C++, .NET; integrates with MATLAB/Simulink
- **Streaming:** Supports Lab Streaming Layer (LSL) for real-time Python/C++ pipelines
- **OS:** Unicorn Suite (recording/filtering UI) requires Windows 10; raw acquisition via C-API/Python works on Linux and macOS

### Recommended filter settings (Unicorn Suite)

- Bandpass: 0.1 Hz – 50 Hz
- Notch: 50 Hz (power line noise)
- OSCAR filter (optional): removes eye blinks and muscle artifacts

## Data

Raw data lives in `data/raw/` as CSV with columns:

```
sys_time, eeg_time, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8
```

Sampled at ~250 Hz. Data pipeline directories:
- `data/raw/` — original immutable recordings
- `data/interim/` — intermediate transformed data
- `data/processed/` — final datasets ready for modeling
- `data/external/` — third-party sources

## Environment Setup

Uses `uv` for environment management, Python 3.12.

```bash
make create_environment   # create .venv with uv
source .venv/bin/activate
make requirements         # uv pip install -r requirements.txt
```

## Common Commands

```bash
make lint     # ruff format --check && ruff check
make format   # ruff check --fix && ruff format (auto-fix)
make clean    # remove compiled Python files
```

## Code Architecture

Source package is `brain_id/` (installed as `-e .` via flit):

- `config.py` — shared variables and paths
- `dataset.py` — data loading/generation
- `features.py` — feature engineering (SSVEP features, frequency-domain transforms)
- `modeling/train.py` — model training
- `modeling/predict.py` — inference
- `plots.py` — visualizations

Notebooks go in `notebooks/` with naming convention `<number>.<minor>-<initials>-<description>.ipynb`.  
Trained models are saved to `models/`.  
Generated reports/figures go to `reports/`.

## Code Style

- Line length: 99 (ruff)
- Import sorting enabled (`ruff.lint` `"I"` rule)
- `brain_id` is treated as first-party for import ordering
