import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import (
    load_audio,
    extract_features,
    spec_augment,
)


def build_cache(df, audio_dir, cache_dir, name):
    """
    Extract and cache spectrogram features for all audio files.

    Features are stored as:
        (N, 3, 128, T)

    where the three channels are:
        0 -> log-mel
        1 -> delta
        2 -> delta-delta
    """

    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(
        cache_dir,
        f"{name}.npy"
    )

    if os.path.exists(cache_file):
        print(f"Loading cache: {name}")
        return np.load(
            cache_file,
            mmap_mode="r"
        )

    print(f"Building cache: {name}")

    data = []

    for _, row in df.iterrows():

        path = os.path.join(
            audio_dir,
            row["id"] + ".wav"
        )

        if not os.path.exists(path):
            continue

        y = load_audio(path)
        feat = extract_features(y)

        data.append(feat)

    data = np.stack(data)

    np.save(
        cache_file,
        data
    )

    print(
        f"Saved {name} cache: {data.shape}"
    )

    return data


class AudioDataset(Dataset):
    """
    Dataset for cached three-channel audio features.

    Training:
        - Applies SpecAugment to the mel channel.
        - Standardizes each channel independently.

    Evaluation:
        - No SpecAugment.
        - Standardizes each channel independently.
    """

    def __init__(
        self,
        features,
        labels=None,
        ids=None,
        audio_dir=None,
        train=True
    ):
        self.features = features
        self.labels = labels
        self.ids = ids
        self.audio_dir = audio_dir
        self.train = train

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        # Copy because cached arrays may be memory-mapped
        feat = self.features[idx].copy()

        # SpecAugment only during training
        if self.train:
            feat[0] = spec_augment(feat[0])

        # Per-channel standardization
        for c in range(feat.shape[0]):

            mean = feat[c].mean()
            std = feat[c].std() + 1e-6

            feat[c] = (
                feat[c] - mean
            ) / std

        x = torch.tensor(
            feat,
            dtype=torch.float32
        )

        if self.labels is not None:

            y = torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )

            return x, y

        return x
