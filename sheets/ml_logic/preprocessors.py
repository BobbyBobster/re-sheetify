import numpy as np
from abc import ABC, abstractmethod

import librosa

from sheets.params import *


class Preprocessor(ABC):
    @abstractmethod
    def compute(self, audio: np.ndarray) -> np.ndarray:
        pass


class CQTPreprocessor:
    def compute(self, audio: np.ndarray) -> np.ndarray:
        cqt = librosa.cqt(
            audio,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=FMIN,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )
        # Convert to dB scale, with CQT we should use amplitude_to_db
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

        # Normalize to [0, 1]
        cqt_min, cqt_max = cqt_db.min(), cqt_db.max()
        if cqt_max > cqt_min:
            cqt_db = (cqt_db - cqt_min) / (cqt_max - cqt_min)
        else:
            cqt_db = np.zeros_like(cqt_db)

        # Add channel dim for Conv2D compatibility → (bins, frames, 1)
        return cqt_db[:, :, np.newaxis].astype(np.float32)


class MELPreprocessor:
    def compute(self, audio: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            n_mels=N_MELS,
        )
        # Convert to dB scale, as we are using the default power=2.0 in calculating
        # the melspectrogram, we should use power_to_db
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_min, mel_max = mel_db.min(), mel_db.max()
        if mel_max > mel_min:
            mel_db = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_db = np.zeros_like(mel_db)

        # Add channel dim for Conv2D compatibility → (n_mels, frames, 1)
        return mel_db[:, :, np.newaxis].astype(np.float32)
