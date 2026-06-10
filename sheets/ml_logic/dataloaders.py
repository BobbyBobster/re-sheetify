import json
from pathlib import Path

import numpy as np
import librosa
import pretty_midi

from typing import Literal

from sheets.ml_logic.preprocessors import Preprocessor, CQTPreprocessor
from sheets.params import *


def get_pairs(
    data_path=DATA_PATH,
    split: SplitType = "train",
    year_limit: int | list[int] | None = None,
    count_limit: int | None = None,
) -> list[dict]:
    """
    Returns list of {audio_path, midi_path, duration} dicts for a split.
    """

    with open(data_path / "maestro-v3.0.0.json", "r") as fp:
        metadata = json.load(fp)

    if year_limit is None:
        year_limit = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
    if isinstance(year_limit, int):
        year_limit = [year_limit]

    # Filter keys on year and split
    year_filt = [k for k, v in metadata["year"].items() if v in year_limit]
    split_filt = [k for k, v in metadata["split"].items() if v == split]
    keys = [k for k in year_filt if k in split_filt]
    # Get filenames for filtered keys
    audio_filenames = [v for k, v in metadata["audio_filename"].items() if k in keys]
    midi_filenames = [v for k, v in metadata["midi_filename"].items() if k in keys]
    durations = [v for k, v in metadata["duration"].items() if k in keys]

    pairs = [
        {
            "audio_path": str(data_path / "mp3s" / a_fname.replace(".wav", ".mp3")),
            "midi_path": str(data_path / "midis" / m_fname),
            "duration": duration,
        }
        for a_fname, m_fname, duration in zip(
            audio_filenames, midi_filenames, durations
        )
    ]

    if count_limit is not None:
        pairs = pairs[:count_limit]

    # TODO: Remove after testing
    if year_limit == [0]:
        pairs = [
            {
                "audio_path": str(data_path / "mp3s" / "midi.mp3"),
                "midi_path": str(data_path / "midis" / "midi.mid"),
                "duration": 26,
            }
        ]

    print(f"✅ [Metadata] {split}: {len(pairs)} files found.")
    return pairs


def _pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def preprocess_spectrogram(
    audio_path: str,
    preprocessor: Preprocessor,
    start_sec,
):
    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=start_sec,
        duration=CLIP_DURATION,
    )
    clip_samples = int(SAMPLE_RATE * CLIP_DURATION)
    audio = _pad_or_trim(audio, clip_samples)
    spectrogram = preprocessor.compute(audio)
    return spectrogram


def _precomputed_path(name: str, pair: dict, start_sec: float) -> Path:
    precomputed_root = DATA_PATH / ("precomputed_" + name.lower())
    clip_i = int(start_sec / CLIP_DURATION)
    # Keep only year and filename
    y_f = Path("/".join(Path(pair["audio_path"]).parts[-2:])).with_suffix("")
    return precomputed_root / y_f.parent / f"{y_f.name}_{clip_i:05d}.npy"


def load_CQT(
    pair: dict,
    start_sec: float,
) -> np.ndarray:
    """Load a (precomputed) CQT spectrogram"""
    cqt_path = _precomputed_path("cqt", pair, start_sec)
    if cqt_path.exists():
        # print(f'🔋 Dataloader: Precomputed CQT found : {pair["audio_path"]=}')
        cqt = np.load(cqt_path).astype(np.float32)
    else:
        print(f'🪫 Dataloader: No precomputed CQT : {pair["audio_path"]=}')
        cqt = preprocess_spectrogram(
            pair["audio_path"],
            preprocessor=CQTPreprocessor(),
            start_sec=start_sec,
        )
    return cqt


def load_MEL(
    pair: dict,
    start_sec: float,
) -> np.ndarray:
    """Load a (precomputed) MEL spectrogram"""
    # TODO: Createt precomputed MELs
    # TODO: Implement loading
    pass


def _pad_or_trim_2d(arr: np.ndarray, length: int, axis: int = 1) -> np.ndarray:
    current = arr.shape[axis]
    if current >= length:
        return np.take(arr, range(length), axis=axis)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, length - current)
    return np.pad(arr, pad_width)


def get_pianoroll(
    midi_path: str,
):
    midi = pretty_midi.PrettyMIDI(midi_path)
    roll = midi.get_piano_roll(fs=PIANO_ROLL_FS)
    return roll


def slice_pianoroll(
    roll: np.ndarray,
    start_sec: float,
) -> np.ndarray:
    frame_start = int(start_sec * PIANO_ROLL_FS)
    frame_end = frame_start + int(CLIP_DURATION * PIANO_ROLL_FS)
    roll = roll[:, frame_start:frame_end]
    roll = roll[PIANO_MIN_PITCH : PIANO_MAX_PITCH + 1, :]
    roll = (roll > 0).astype(np.float32)
    target_frames = int(CLIP_DURATION * PIANO_ROLL_FS)
    roll = _pad_or_trim_2d(roll, target_frames, axis=1)
    return roll


def create_pianoroll(
    midi_path: str,
    start_sec: float,
) -> np.ndarray:
    roll = get_pianoroll(midi_path)
    roll = slice_pianoroll(roll, start_sec)
    return roll


def load_roll(
    pair: dict,
    start_sec: float,
) -> np.ndarray:
    """Load a (precomputed) pianoroll."""
    roll_path = _precomputed_path("roll", pair, start_sec)
    if roll_path.exists():
        # print(f'🔋 Dataloader: Precomputed pianrolls found : {pair["midi_path"]=}')
        roll = np.load(roll_path).astype(np.float32)
    else:
        print(f'🪫 Dataloader: No precomputed pianorolls : {pair["midi_path"]=}')
        roll = create_pianoroll(midi_path=pair["midi_path"], start_sec=start_sec)
    return roll
