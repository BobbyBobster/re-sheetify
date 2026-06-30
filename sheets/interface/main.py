import tensorflow as tf
import time
import logging

from sheets.config import setup_logging
from pathlib import Path

from sheets.ml_logic.dataset_builder import build_dataset
import sheets.models.basic_model as basic_model
import sheets.models.onf_model as onf_model
from sheets.params import *

setup_logging()
logger = logging.getLogger(__name__)

DATASET_ROOT: Path = Path("./dataset_cache")


def save_dataset(
    model_type: ModelType = "basic",
    year_limit: list[int] | None = None,
    count_limit: int | None = None,
    batch_size=32,
    patience=10,
    epochs=200,
):
    # Run dataset creation
    # Save to GCS, Service Account needs Storage Object Admin on bucket
    # gcs_output_path = "gs://your-bucket-name/preprocessed_dataset"
    # logger.info(f"⏳ Saving dataset to {gcs_output_path}...")
    # tf.data.Dataset.save(dataset, gcs_output_path)
    # logger.info("✅ Dataset successfully saved!")

    train_ds = build_dataset(
        model_type=model_type,
        split="train",
        year_limit=year_limit,
        count_limit=count_limit,
        batch_size=batch_size,
    )
    train_ds_path = DATASET_ROOT / "train_ds_dir"
    train_ds.save(str(train_ds_path))


def train(
    model_type: ModelType = "basic",
    year_limit: list[int] | None = None,
    count_limit: int | None = None,
    batch_size=32,
    patience=10,
    epochs=200,
):
    logger.info(f"📌 Initializing {model_type} model type.")
    if model_type == "basic":
        model = basic_model.initialize_model()
        model = basic_model.compile_model(model)
    elif model_type == "onf":
        model = onf_model.initialize_model()
        model = onf_model.compile_model(model)
    else:
        logger.error("❌ No model type to train selected. Exiting.")
        raise SystemExit

    train_ds_path = Path(DATASET_ROOT) / "train_ds_dir"
    if train_ds_path.exists():
        logger.info("🔋 Saved dataset found: train_ds")
        train_ds = tf.data.Dataset.load(str(train_ds_path))
    else:
        logger.info("🪫 No dataset found: train_ds")
        logger.info("📌 Initializing 'train' dataset (data will be loaded lazily).")
        train_ds = build_dataset(
            model_type=model_type,
            split="train",
            year_limit=year_limit,
            count_limit=count_limit,
            batch_size=batch_size,
        )

    val_ds_path = Path(DATASET_ROOT) / "val_ds_dir"
    if val_ds_path.exists():
        logger.info("🔋 Saved dataset found: val_ds")
        val_ds = tf.data.Dataset.load(str(val_ds_path))
    else:
        logger.info("🪫 No dataset found: val_ds")
        logger.info(
            "📌 Initializing 'validation' dataset (data will be loaded lazily)."
        )
        val_ds = build_dataset(
            model_type=model_type,
            split="validation",
            year_limit=year_limit,
            count_limit=count_limit,
            batch_size=batch_size,
        )

    if year_limit is None:
        year_limit = "all"  # type: ignore

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = (
        f"./models/{model_type}_y{''.join(map(str, year_limit))}_{timestamp}.keras"  # type: ignore
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_best_only=True
    )
    logger.info(f"📌 Saving checkpoints at '{checkpoint_filepath}'")

    earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience)

    history = model.fit(  # noqa: F841
        x=train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[earlystopping_cb, checkpoint_cb],
        verbose=2,  # type: ignore
    )

    # Load the best weights
    model = tf.keras.models.load_model(checkpoint_filepath, safe_mode=False)

    # Save model locally
    if MODEL_TARGET == "local":
        model_path = os.path.join("./models", f"{timestamp}.keras")
        model.save(model_path)

    logger.info("✅ train() done \n")


def predict():
    pass
