# Spoken Digit Recognition

Deep learning system for recognizing spoken digits (0–9) from short audio clips using a Residual CNN–BiGRU architecture with Squeeze-and-Excitation attention.

## Overview

This project was developed for the Kaggle Spoken Digit Recognition competition. The task is to classify short audio recordings of spoken digits from 0 to 9, including recordings from unseen speakers and noisy/augmented audio.

## Model Pipeline

Audio Waveform
→ Mel-Spectrogram + Δ + ΔΔ
→ Residual CNN
→ Squeeze-and-Excitation Attention
→ Frequency Pooling
→ Bidirectional GRU
→ Fully Connected Layer
→ Digit Prediction (0–9)

## Dataset

- Training samples: 37,800
- Test samples: 16,000
- Audio format: WAV
- Sampling rate: 16 kHz
- Duration: 1 second
- Classes: 0–9

## Feature Extraction

Each audio sample is converted into a three-channel spectrogram:

- Log-Mel spectrogram with 128 Mel bins
- First-order delta (Δ)
- Second-order delta (ΔΔ)

The spectrogram features are standardized before being passed to the neural network.

## Data Augmentation

The training pipeline uses multiple augmentation techniques:

- Gaussian noise
- Time stretching
- Pitch shifting
- Random gain
- Time cutout
- SpecAugment
- Mixup

These augmentations improve robustness to speaker and recording variations.

## Model Architecture

The model contains:

- Residual CNN backbone
- Squeeze-and-Excitation channel attention
- Frequency pooling
- 2-layer Bidirectional GRU
- Fully connected classification head
- Approximately 6.2M trainable parameters

## Training

- Optimizer: AdamW
- Initial learning rate: 1e-3
- Weight decay: 1e-4
- Loss: Label-smoothed cross entropy
- Mixup: Beta(0.4, 0.4)
- Learning-rate scheduler: Cosine annealing
- Gradient clipping: 5.0
- Training epochs: 20
- Batch size: 64

## Results

| Metric | Result |
|---|---:|
| Private Kaggle F1 Score | **0.99285** |
| Team Rank | **18 / 25** |
| Training Macro F1 | **0.9999** |
| Train Samples | **37,800** |
| Test Samples | **16,000** |

The model achieved a **0.99285 F1 score on the private Kaggle leaderboard**, placing the team **18th among 25 teams**.

The reported training macro F1 of 0.9999 was calculated on the full training split and should not be interpreted as a held-out validation score.

## Kaggle Competition

[View Kaggle Competition]([YOUR_KAGGLE_LINK](https://www.kaggle.com/competitions/digitrecognition-ee708))

## Technologies

- Python
- PyTorch
- Librosa
- NumPy
- Pandas
- Scikit-learn

## Repository

The original Kaggle training notebook is included in this repository.
