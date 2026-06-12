import keras

from sheets.models.losses_metrics import *
from sheets.params import *


def initialize_model(n_kernel=32) -> keras.Model:
    inputs = keras.Input(shape=(N_BINS, N_FRAMES, 1))

    bin_dim = N_BINS
    frame_dim = N_FRAMES

    # --- CNN Block ---
    x = keras.layers.Conv2D(n_kernel, (3, 3), activation="relu", padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    n_kernel *= 2
    x = keras.layers.Conv2D(n_kernel, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    bin_dim //= 2
    frame_dim //= 2
    x = keras.layers.Dropout(0.3)(x)

    n_kernel *= 2
    x = keras.layers.Conv2D(n_kernel, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    bin_dim //= 2
    frame_dim //= 2
    # Tensor output shape here is (None, 21, 78, 256)
    x = keras.layers.Dropout(0.3)(x)

    # --- Collapse PITCH ---
    x = keras.layers.Lambda(
        lambda t: keras.ops.mean(t, axis=1),
        output_shape=(frame_dim, n_kernel),
    )(x)

    # --- Add Positional Tracking for the Transformer ---
    x = PositionalEmbeddingAdder(input_dim=frame_dim, output_dim=n_kernel)(x)

    # --- Transformer Block ---
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = keras.layers.LayerNormalization()(x)

    # --- Learn the Time Mapping (No Blur) ---
    # Dynamically maps 78 audio steps up to 1000 piano-roll time targets.
    n_time = int(CLIP_DURATION * PIANO_ROLL_FS)
    x = keras.layers.Permute((2, 1))(x)  # Shape: (None, 256, 78)
    x = keras.layers.Dense(n_time // 4)(x)  # Shape: (None, 256, 1000)
    x = keras.layers.Permute((2, 1))(x)  # Shape: (None, 1000, 256)
    # Scale up from 250 to 1000 frames using interpolation
    x = keras.layers.UpSampling1D(size=4)(x)  # Shape: (None, 1000, 256)

    # --- Output Block ---
    # Map 256 features directly to your 88 individual keys across every step.
    x = keras.layers.Dense(N_PIANO_KEYS)(x)  # Shape: (None, 1000, 88)

    # Rearrange dimensions to deliver your target formatting: (None, 88, 1000)
    outputs = keras.layers.Permute((2, 1))(x)
    outputs = keras.layers.Activation("sigmoid")(outputs)

    model = keras.Model(inputs, outputs)

    print("✅ Model initialized")

    return model


def compile_model(
    model: keras.Model,
    learning_rate=LEARNING_RATE,
) -> keras.Model:
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


def evaluate_model():
    pass


def predict():
    pass
