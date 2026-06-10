import keras

from sheets.params import *


@keras.saving.register_keras_serializable()
class PositionalEmbeddingAdder(keras.layers.Layer):
    """
    Custom layer which adds positional information.
    """

    def __init__(self, input_dim=400, output_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Instantiate inside __init__ so Keras tracks the weights natively
        self.pos_embedding_layer = keras.layers.Embedding(
            input_dim=self.input_dim, output_dim=self.output_dim
        )

    def call(self, inputs):
        # Dynamically calculate positions using keras.ops
        seq_len = keras.ops.shape(inputs)[1]
        positions = keras.ops.arange(0, seq_len, 1)

        # Add the positional embeddings to your inputs
        return inputs + self.pos_embedding_layer(positions)

    def get_config(self):
        # This tells Keras how to recreate the layer when loading
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable()
def flattened_fbeta_score(y_true, y_pred):
    y_true_flat = keras.ops.reshape(y_true, [-1, 88 * 1000])
    y_pred_flat = keras.ops.reshape(y_pred, [-1, 88 * 1000])
    return keras.metrics.binary_accuracy(y_true_flat, y_pred_flat)


flattened_fbeta_score.name = "fbeta"


@keras.saving.register_keras_serializable()
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """
    Custom loss function which penalizes false negatives.
    This will help with our class imbalance. There are way more positions where
    a note is _not_ played than played.
    Set penalty with `pos_weight`.
    If there are about 10x more zeros than ones, set `pos_weight=10.0`.
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        reduction: str = "sum_over_batch_size",
        name: str = "weighted_bce",
    ):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        y_true = keras.ops.cast(y_true, y_pred.dtype)

        # Prevent log(0) or log(1) numerical instability
        epsilon = keras.config.epsilon()
        y_pred = keras.ops.clip(y_pred, epsilon, 1.0 - epsilon)

        # Standard weighted BCE math applied directly to probabilities
        loss = -(
            self.pos_weight * y_true * keras.ops.log(y_pred)  # type: ignore
            + (1.0 - y_true) * keras.ops.log(1.0 - y_pred)  # type: ignore
        )

        # Average over the last dimension (features/classes)
        return keras.ops.mean(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"pos_weight": self.pos_weight})
        return config


@keras.saving.register_keras_serializable()
class FlattenedBinaryAccuracy(keras.metrics.Metric):
    def __init__(self, name="flattened_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.accuracy_sum = self.add_weight(name="accuracy_sum", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        area = int(N_PIANO_KEYS * (CLIP_DURATION * PIANO_ROLL_FS))

        y_true_flat = keras.ops.reshape(y_true, [-1, area])
        y_pred_flat = keras.ops.reshape(y_pred, [-1, area])

        acc = keras.metrics.binary_accuracy(y_true_flat, y_pred_flat)

        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, acc.dtype)
            acc = acc * sample_weight  # type: ignore
        else:
            weight = keras.ops.cast(1 / area, acc.dtype)
            acc = acc * weight  # type: ignore

        self.accuracy_sum.assign_add(keras.ops.sum(acc))
        self.total_samples.assign_add(
            keras.ops.cast(keras.ops.shape(acc)[0], self.total_samples.dtype)
        )

    def result(self):
        return self.accuracy_sum / (self.total_samples + keras.config.epsilon())

    def reset_state(self):
        self.accuracy_sum.assign(0.0)
        self.total_samples.assign(0.0)
