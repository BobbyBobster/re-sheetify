import keras

from sheets.models.losses_metrics import *
from sheets.params import *


def conv_stack(x):
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling2D((1, 2))(x)
    x = keras.layers.Dropout(0.3)(x)

    return x


def squeeze_freq(x):
    x = keras.layers.Permute((2, 1, 3))(x)
    shape = x.shape
    return keras.layers.Reshape((shape[1], shape[2] * shape[3]))(x)


def map_to_target_time(x, n_time, name):
    x = keras.layers.Permute((2, 1))(x)
    x = keras.layers.Dense(n_time)(x)
    x = keras.layers.Activation("sigmoid", name=name)(x)

    return x


def initialize_model(
    n_bins=N_BINS,
    n_frames=N_FRAMES,
    n_keys=N_PIANO_KEYS,
    n_time=int(CLIP_DURATION * PIANO_ROLL_FS),
) -> keras.Model:
    inputs = keras.Input(shape=(n_bins, n_frames, 1))

    onset_x = conv_stack(inputs)
    onset_x = squeeze_freq(onset_x)
    onset_x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(
        onset_x
    )

    onset_seq = keras.layers.Dense(
        n_keys,
        activation="sigmoid",
    )(onset_x)

    frame_x = conv_stack(inputs)
    frame_x = squeeze_freq(frame_x)

    frame_seq = keras.layers.Dense(
        n_keys,
        activation="sigmoid",
    )(frame_x)

    combined = keras.layers.Concatenate()([frame_seq, onset_seq])
    combined = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True)
    )(combined)

    frame_seq = keras.layers.Dense(n_keys)(combined)

    onset_output = map_to_target_time(onset_seq, n_time, "onset")
    frame_output = map_to_target_time(frame_seq, n_time, "frame")

    model = keras.Model(
        inputs=inputs,
        outputs=[onset_output, frame_output],
    )

    print("✅ Onsets & Frames model initialized")
    return model


def compile_model(
    model: keras.Model,
    learning_rate=6e-4,
) -> keras.Model:

    fba = FlattenedBinaryAccuracy()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        # TODO: Test with Weighted BCE
        loss={
            "onset": "binary_crossentropy",
            "frame": "binary_crossentropy",
        },
        loss_weights={
            "onset": 1.0,
            "frame": 1.0,
        },
        # TODO: Add Flattened F score metrics
        metrics={
            "onset": [flattened_fbeta_score, fba],
            "frame": [flattened_fbeta_score, fba],
        },
    )

    print("✅ Model compiled")
    return model


def train_model():
    pass
