import keras

from sheets.models.losses_metrics import *
from sheets.params import *


def conv_stack(x, n_kernel=64):
    x = keras.layers.Conv2D(n_kernel, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(n_kernel * 2, (3, 3), activation="relu", padding="same")(x)
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


def initialize_model(n_kernel=64) -> keras.Model:
    inputs = keras.Input(shape=(N_BINS, N_FRAMES, 1))

    onset_x = conv_stack(inputs, n_kernel)
    onset_x = squeeze_freq(onset_x)
    onset_x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(
        onset_x
    )

    onset_seq = keras.layers.Dense(
        N_PIANO_KEYS,
        activation="sigmoid",
    )(onset_x)

    frame_x = conv_stack(inputs, n_kernel)
    frame_x = squeeze_freq(frame_x)

    frame_seq = keras.layers.Dense(
        N_PIANO_KEYS,
        activation="sigmoid",
    )(frame_x)

    combined = keras.layers.Concatenate()([frame_seq, onset_seq])
    combined = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True)
    )(combined)

    frame_seq = keras.layers.Dense(N_PIANO_KEYS)(combined)

    n_time = int(CLIP_DURATION * PIANO_ROLL_FS)
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
    learning_rate=LEARNING_RATE,
) -> keras.Model:
    onset_wbc = WeightedBinaryCrossentropy(pos_weight=4000)
    frame_wbc = WeightedBinaryCrossentropy(pos_weight=200)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    fba = FlattenedBinaryAccuracy()

    model.compile(
        optimizer=optimizer,
        loss={
            "onset": onset_wbc,
            "frame": frame_wbc,
        },
        loss_weights={
            "onset": 1.0,
            "frame": 1.0,
        },
        metrics={
            "onset": [flattened_fbeta_score, fba],
            "frame": [flattened_fbeta_score, fba],
        },
    )

    print("✅ Model compiled")
    return model


def train_model():
    pass


def evaluate_model():
    pass


def predict():
    pass
