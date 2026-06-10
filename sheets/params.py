import os
from pathlib import Path
from typing import TypeAlias, Literal

### CONSTANTS ###
DATA_PATH = Path("./data")
CACHE_PATH = Path("./.sheetify_cache")
SAMPLE_RATE = 16_000  # Hz
CLIP_DURATION = 10.0  # seconds
HOP_LENGTH = 512  # ~32ms at 16kHz
N_BINS = 84  # 7 octaves × 12 semitones (full piano range)
BINS_PER_OCTAVE = 12
N_FRAMES = 313
N_MELS = 84  # changed from 128 → must match model input (84, 313, 1)
N_FFT = 2048 # Length of fft window
FMIN = 27.5  # Hz — lowest piano key
PIANO_ROLL_FS = 100  # frames per second — matches CQT pipeline
PIANO_MIN_PITCH = 21  # MIDI A0
PIANO_MAX_PITCH = 108  # MIDI C8
N_PIANO_KEYS = PIANO_MAX_PITCH - PIANO_MIN_PITCH + 1  # 88
LEARNING_RATE = 0.0005


### TYPES ###
ModelType: TypeAlias = Literal["basic", "onf"]
SpectrogramType: TypeAlias = Literal["cqt", "mel"]
SplitType: TypeAlias = Literal["train", "validation", "test"]


### VARIABLES ###
MODEL_TARGET = os.environ.get("MODEL_TARGET")
# Local data storage
# ML_DIR = os.environ.get("ML_DIR", "./mlops")
# LOCAL_DATA_PATH = Path(ML_DIR) / "data"
# LOCAL_REGISTRY_PATH =  Path(ML_DIR) / "training_outputs"
# GCP Project
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
# Cloud Storage
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# Compute Engine
INSTANCE = os.environ.get("INSTANCE")
# MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# Prefect
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
# Trainer image
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

### VALIDATIONS ###
env_valid_options = dict(
    MODEL_TARGET=["local", "gcs", "mlflow"],
)


def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(
            f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}"
        )


for env, valid_options in env_valid_options.items():
    pass
    # validate_env_value(env, valid_options)
