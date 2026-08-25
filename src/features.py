import random
import numpy as np
import librosa


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SR = 16000
SAMPLES = SR
N_MELS = 128


# ─────────────────────────────────────────────
# Waveform Augmentation
# ─────────────────────────────────────────────

def add_noise(y, min_snr=15, max_snr=40):
    """Add Gaussian noise at a random SNR."""
    snr_db = random.uniform(min_snr, max_snr)

    rms_sig = np.sqrt(np.mean(y ** 2)) + 1e-9
    rms_noise = rms_sig / (10 ** (snr_db / 20))

    noise = np.random.randn(len(y)) * rms_noise

    return (y + noise).astype(np.float32)


def time_stretch(y, min_rate=0.85, max_rate=1.15):
    """Speed up or slow down without changing pitch."""
    rate = random.uniform(min_rate, max_rate)

    y_stretched = librosa.effects.time_stretch(
        y,
        rate=rate
    )

    if len(y_stretched) < SAMPLES:
        y_stretched = np.pad(
            y_stretched,
            (0, SAMPLES - len(y_stretched))
        )
    else:
        y_stretched = y_stretched[:SAMPLES]

    return y_stretched


def pitch_shift(y, min_steps=-3, max_steps=3):
    """Shift pitch by a random number of semitones."""
    steps = random.uniform(min_steps, max_steps)

    return librosa.effects.pitch_shift(
        y,
        sr=SR,
        n_steps=steps
    )


def random_gain(y, min_db=-6, max_db=6):
    """Apply random gain in decibels."""
    gain = 10 ** (
        random.uniform(min_db, max_db) / 20
    )

    return np.clip(
        y * gain,
        -1.0,
        1.0
    )


def time_cutout(y, max_fraction=0.2):
    """Zero out a random contiguous segment."""
    y = y.copy()

    n = int(
        len(y) *
        random.uniform(0.05, max_fraction)
    )

    start = random.randint(
        0,
        len(y) - n
    )

    y[start:start + n] = 0.0

    return y


def augment_waveform(y):
    """Apply a random subset of waveform augmentations."""

    if random.random() < 0.5:
        y = add_noise(y)

    if random.random() < 0.4:
        y = time_stretch(y)

    if random.random() < 0.4:
        y = pitch_shift(y)

    if random.random() < 0.4:
        y = random_gain(y)

    if random.random() < 0.3:
        y = time_cutout(y)

    return y


# ─────────────────────────────────────────────
# Spectrogram Augmentation
# ─────────────────────────────────────────────

def spec_augment(
    mel,
    n_freq_masks=2,
    n_time_masks=2,
    freq_width=15,
    time_width=20
):
    """
    Apply SpecAugment to a mel spectrogram.

    Frequency masking removes random mel bands.
    Time masking removes random time windows.
    """

    mel = mel.copy()

    n_freq, n_time = mel.shape

    # Frequency masking
    for _ in range(n_freq_masks):
        f = random.randint(0, freq_width)
        f0 = random.randint(0, n_freq - f)

        mel[f0:f0 + f, :] = 0

    # Time masking
    for _ in range(n_time_masks):
        t = random.randint(0, time_width)
        t0 = random.randint(0, n_time - t)

        mel[:, t0:t0 + t] = 0

    return mel


# ─────────────────────────────────────────────
# Audio Loading
# ─────────────────────────────────────────────

def load_audio(path):
    """
    Load mono audio at 16 kHz and make it
    exactly one second long.
    """

    y, _ = librosa.load(
        path,
        sr=SR
    )

    if len(y) < SAMPLES:
        y = np.pad(
            y,
            (0, SAMPLES - len(y))
        )
    else:
        y = y[:SAMPLES]

    return y.astype(np.float32)


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

def extract_features(y):
    """
    Convert a waveform into a 3-channel
    mel-spectrogram representation.

    Channel 0 = log-mel spectrogram
    Channel 1 = first-order delta
    Channel 2 = second-order delta

    Returns:
        (3, 128, T)
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=160
    )

    mel_db = librosa.power_to_db(
        mel
    ).astype(np.float32)

    delta = librosa.feature.delta(
        mel_db,
        order=1
    )

    delta2 = librosa.feature.delta(
        mel_db,
        order=2
    )

    return np.stack(
        [mel_db, delta, delta2],
        axis=0
    )
