# HeartMRI-FL

A starter project for federated learning with heart MRI data.

## Overview

This repository provides a Python scaffold for privacy-preserving healthcare anomaly detection using:

- Federated Learning across hospitals
- Client-level Differential Privacy
- Adaptive intermediary sub-client splitting
- Autoencoder-based anomaly detection

## Structure

- `src/heartmri_fl/` - core library code
- `notebooks/` - example experiment notebook
- `tests/` - unit tests
- `requirements.txt` - Python dependencies

## Getting Started

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - Windows (Command Prompt):
     ```bash
     .venv\Scripts\activate
     ```
   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Optionally install the package in editable mode:
   ```bash
   pip install -e .
   ```
5. Run the sample federated entrypoint:
   ```bash
   python -m heartmri_fl
   ```
6. Run with a custom configuration:
   ```bash
   python -m heartmri_fl --data-dir ./data --clients 3 --subclients-per-hospital 2 --rounds 5 --task autoencoder --noise-multiplier 1.0
   ```

## Privacy and Anomaly Detection

- The project supports client-level differential privacy by clipping client updates and adding Gaussian noise.
- The adaptive intermediary strategy increases the number of virtual sub-clients per hospital, which helps reduce the impact of DP noise.
- The autoencoder learns normal MRI patterns and detects anomalies via reconstruction error.

## Data

- If `./data` contains DICOM, NIfTI, or JPG files organized by hospital folder, the project loads real medical images.
- Expected directory layout:
  ```
  ./data/
    hospital_0/
      scan1.dcm
      scan2.jpg
      patientA.nii
    hospital_1/
      scan1.dcm
      scan2.jpg
      patientB.nii.gz
  ```
- Each hospital folder is treated as one client; `--subclients-per-hospital` further splits its scans into virtual subclients.
- Supported file formats: `.dcm`, `.nii`, `.nii.gz`, `.jpg`, `.jpeg`.
- If no valid medical data is found, the project falls back to synthetic non-IID hospital data with anomalies.

## Experiment Notebook

Open `notebooks/heartmri_fl_experiment.ipynb` to run a federated example and inspect results.

## Next Improvements

- Add labeled anomaly datasets and clinical evaluation metrics
- Implement formal privacy accounting for `ε`
- Add segmentation or supervised clinical tasks
- Extend evaluation for non-IID hospital distributions
