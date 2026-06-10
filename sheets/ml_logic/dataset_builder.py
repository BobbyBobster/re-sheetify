import math

import tensorflow as tf
import numpy as np

from typing import Literal

import sheets.ml_logic.dataloaders as dl
from sheets.params import *


def create_onf_roll(
    piano_roll: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_roll = piano_roll.astype(np.float32)

    padded = np.pad(frame_roll, ((0, 0), (1, 0)), mode="constant")

    # Keep only the first instance of a note start
    onset_roll = (padded[:, 1:] > 0) & (padded[:, :-1] == 0)
    onset_roll = onset_roll.astype(np.float32)

    return onset_roll, frame_roll


def build_dataset(
    model_type: ModelType = "basic",
    spectrogram_type: SpectrogramType = "cqt",
    split: SplitType = "train",
    year_limit: int | list[int] | None = None,
    count_limit: int | None = None,
    batch_size=16,
    shuffle_buffer: int = 200,
    prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    pairs = dl.get_pairs(split=split, year_limit=year_limit, count_limit=count_limit)

    def generator():
        for pair in pairs:
            num_splits = math.floor(pair["duration"] / CLIP_DURATION)
            split_starts = [CLIP_DURATION * idx for idx in range(num_splits)]
            for split_start in split_starts:
                try:
                    roll = dl.load_roll(pair, split_start)
                    if model_type == "onf":
                        roll = create_onf_roll(roll)

                    if spectrogram_type == "cqt":
                        spectrogram = dl.load_CQT(pair, split_start)
                    elif spectrogram_type == "mel":
                        spectrogram = dl.load_MEL(pair, split_start)

                    yield spectrogram, roll
                except Exception as e:
                    print(
                        f"[Warning] Skipping {pair['audio_path']} at {split_start} seconds: {e}"
                    )
                    continue

    # Infer output shapes
    n_frames = int(CLIP_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1
    roll_frames = int(CLIP_DURATION * PIANO_ROLL_FS)

    if model_type == "basic":
        output_signature = (
            tf.TensorSpec(shape=(N_BINS, n_frames, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(N_PIANO_KEYS, roll_frames), dtype=tf.float32),
        )
    elif model_type == "onf":
        output_signature = (
            tf.TensorSpec(
                shape=(N_BINS, n_frames, 1),
                dtype=tf.float32,
                name="feat_spectrogram",
            ),
            (
                tf.TensorSpec(
                    shape=(N_PIANO_KEYS, roll_frames),
                    dtype=tf.float32,
                    name="target_onset_roll",
                ),
                tf.TensorSpec(
                    shape=(N_PIANO_KEYS, roll_frames),
                    dtype=tf.float32,
                    name="target_frame_roll",
                ),
            ),
        )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    )

    cache_dir = CACHE_PATH
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = Path(f"cache_{split}_{year_limit}_{model_type}_{spectrogram_type}")
    dataset = dataset.cache(str(cache_dir / cache_name))

    # if split == "train":
    # dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,  # TODO: Investigate whether this should be True or False
        # When True, the model.fit on a basic model raises a math.domain
        # exception on a logarithm. The cause of this is unknown
    ).prefetch(prefetch)

    return dataset
