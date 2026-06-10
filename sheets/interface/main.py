import tensorflow as tf
import numpy as np
import time

from sheets.ml_logic.dataset_builder import build_dataset
import sheets.models.basic_model as basic_model
import sheets.models.onf_model as onf_model
from sheets.params import *


def create_cache():
    # Run dataset creation
    # Save to GCS, Service Account needs Storage Object Admin on bucket
    # gcs_output_path = "gs://your-bucket-name/preprocessed_dataset"
    # print(f"⏳ Saving dataset to {gcs_output_path}...")
    # tf.data.Dataset.save(dataset, gcs_output_path)
    # print("✅ Dataset successfully saved!")
    pass


def train(
    model_type: ModelType = "basic",
    year_limit: list[int] | None = None,
    count_limit: int | None = None,
    batch_size=32,
    patience=10,
    epochs=200,
):
    print(f"📌 Initializing {model_type} model type.")
    if model_type == "basic":
        model = basic_model.initialize_model()
        model = basic_model.compile_model(model)
    elif model_type == "onf":
        model = onf_model.initialize_model()
        model = onf_model.compile_model(model)
    else:
        print("❌ No model type to train selected. Exiting.")
        raise SystemExit

    print(f"📌 Initializing 'train' dataset (data will be loaded lazily).")
    train_ds = build_dataset(
        model_type=model_type,
        split="train",
        year_limit=year_limit,
        count_limit=count_limit,
        batch_size=batch_size,
    )
    print(f"📌 Initializing 'validation' dataset (data will be loaded lazily).")
    val_ds = build_dataset(
        model_type=model_type,
        split="validation",
        year_limit=year_limit,
        count_limit=count_limit,
        batch_size=batch_size,
    )

    if year_limit is None:
        year_limit = "all"

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = (
        f"./models/{model_type}_y{''.join(map(str, year_limit))}_{timestamp}.keras"
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_best_only=True
    )
    print(f"📌 Saving checkpoints at '{checkpoint_filepath}'")

    earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience)

    history = model.fit(
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

    fbeta = np.min(history.history.get("fbeta", 0))

    print("✅ train() done \n")


def predict():
    pass
