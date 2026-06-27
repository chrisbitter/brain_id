# brain_id

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         brain_id and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── brain_id   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes brain_id a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## EEG Device
# Context: g.tec Unicorn Hybrid Black
 
### Overview
The **g.tec Unicorn Hybrid Black** is a wireless 8-channel EEG (Electroencephalography) headset designed for Brain-Computer Interface (BCI) development, academic research, and rapid prototyping.
 
### Technical Specifications
* **Channels:** 8 EEG channels, 3 accelerometer channels, 3 gyroscope channels
* **Signal Quality:** 24-bit resolution, 250 Hz sampling rate
* **Electrodes:** Hybrid (can be used "dry" or "wet" with conductive gel)
* **Connectivity:** Bluetooth
 
### Software & Ecosystem
* **APIs:** Python, C, C++, .NET
* **Integrations:** MATLAB, Simulink
* **OS Compatibility:** The official "Unicorn Suite" (for recording/filtering) requires Windows 10. However, raw data acquisition and processing via C-API and Python bindings are fully functional on Linux and macOS.
* **Streaming:** Supports Lab Streaming Layer (LSL) for time-synchronized, network-based data streaming directly into Python/C++ Machine Learning pipelines.
 
### Typical Use Cases
* **BCI Paradigms:** Motor Imagery, P300 Speller, SSVEP (Steady State Visually Evoked Potentials)
* **Machine Learning:** Real-time brainwave streaming into custom training and inference pipelines
* **Neurofeedback:** Cognitive state evaluation and external device control

## Importan Infos
To collect EEG data while playing the game, follow these steps:

1. Set up the EEG cap:
    - Ensure the cap is properly fitted and all electrodes are in contact with your scalp.
    - Turn the cap on.

1. Launch the Unicorn Suite software and connect to your EEG cap.

1. Select the filter settings in the Unicorn Suite - this is a recommendation:
    - Use a bandpass filter between 0.1 Hz and 50 Hz.
    - Use a notch filter at 50 Hz to remove power line noise.
    - use the OSCAR filter if you want to remove eye blinks and muscle artifacts.


# Main Idea 
We figured out that EEG on this device only works with strong signals but then still not that good and it differs from person to person. Now we want to exploit that weakness and use the uniqueness as a authentication method to login in a Device by measuring EEG signals when looking at some kind of Steady-State Visual Evoked Potential also for different rates. 

We record at 5, 10 adn 15 Hz