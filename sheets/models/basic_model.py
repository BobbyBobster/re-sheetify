import keras

from sheets.models.losses_metrics import *
from sheets.params import *


def initialize_model(
    n_bins=N_BINS,
    n_frames=N_FRAMES,
    n_keys=N_PIANO_KEYS,
    n_time=int(CLIP_DURATION * PIANO_ROLL_FS),
) -> keras.Model:
    inputs = keras.Input(shape=(n_bins, n_frames, 1))

    # --- CNN Block ---
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    # Tensor output shape here is (None, 21, 78, 256)
    x = keras.layers.Dropout(0.3)(x)

    # --- Collapse PITCH ---
    x = keras.layers.Lambda(
        lambda t: keras.ops.mean(t, axis=1),
        output_shape=(78, 128),
    )(x)

    # --- Add Positional Tracking for the Transformer ---
    x = PositionalEmbeddingAdder(input_dim=400, output_dim=128)(x)

    # --- Transformer Block ---
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = keras.layers.LayerNormalization()(x)

    # --- Learn the Time Mapping (No Blur) ---
    # Dynamically maps 78 audio steps up to 1000 piano-roll time targets.
    x = keras.layers.Permute((2, 1))(x)  # Shape: (None, 256, 78)
    x = keras.layers.Dense(n_time // 4)(x)  # Shape: (None, 256, 1000)
    x = keras.layers.Permute((2, 1))(x)  # Shape: (None, 1000, 256)
    # Scale up from 250 to 1000 frames using interpolation
    x = keras.layers.UpSampling1D(size=4)(x)  # Shape: (None, 1000, 256)

    # --- Output Block ---
    # Map 256 features directly to your 88 individual keys across every step.
    x = keras.layers.Dense(n_keys)(x)  # Shape: (None, 1000, 88)

    # Rearrange dimensions to deliver your target formatting: (None, 88, 1000)
    outputs = keras.layers.Permute((2, 1))(x)
    outputs = keras.layers.Activation("sigmoid")(outputs)

    model = keras.Model(inputs, outputs)

    print("✅ Model initialized")

    return model


def compile_model(model: keras.Model, learning_rate=LEARNING_RATE) -> keras.Model:
    wbc = WeightedBinaryCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    fba = FlattenedBinaryAccuracy()

    model.compile(
        loss=wbc,
        optimizer=optimizer,
        metrics=[flattened_fbeta_score, fba],
    )

    print("✅ Model compiled")

    return model


def train_model():
    pass
