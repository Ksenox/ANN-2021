from typing import Tuple

import numpy as np
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model


def generate_data(count: int) -> Tuple[np.array, np.array]:
    x_array = np.random.normal(0, 10, count)
    e_array = np.random.normal(0, 0.3, count)

    functions = [
        lambda x, e: np.power(x, 2) + e,
        lambda x, e: np.sin(x / 2) + e,
        lambda x, e: np.cos(2 * x) + e,
        lambda x, e: x - 3 + e,
        lambda x, e: np.abs(x) + e,
        lambda x, e: np.power(x, 3) / 4 + e,
    ]
    label_function = lambda x, e: -x + e

    data = []
    labels = []
    for x, e in zip(x_array, e_array):
        data.append([fun(x, e) for fun in functions])
        labels.append([label_function(x, e)])

    data = np.array(data)
    labels = np.array(labels)

    mean = data.mean(axis=0)
    std = data.std(axis=0)

    data = (data - mean) / std

    return data, labels


class LayerFactory:
    def __init__(self):
        self.factory_map = {
            "encoder": LayerFactory._create_encoder_layer,
            "decoder": LayerFactory._create_decoder_layer,
            "regression": LayerFactory._create_regression_layer
        }

    @staticmethod
    def _create_encoder_layer(input_layer: Layer) -> Layer:
        layer = Dense(20, activation="relu")(input_layer)
        layer = Dense(10, activation="relu")(layer)
        layer = Dense(3, activation="relu", name="encoder")(layer)
        return layer

    @staticmethod
    def _create_decoder_layer(input_layer: Layer) -> Layer:
        layer = Dense(10, activation="relu")(input_layer)
        layer = Dense(20, activation="relu")(layer)
        layer = Dense(6, activation="relu", name="decoder")(layer)
        return layer

    @staticmethod
    def _create_regression_layer(input_layer: Layer) -> Layer:
        layer = Dense(100, activation="relu")(input_layer)
        layer = Dense(50, activation="relu")(layer)
        layer = Dense(25, activation="relu")(layer)
        layer = Dense(1, name="regression")(layer)
        return layer

    def create(self, layer_name: str):
        return self.factory_map[layer_name]


def create_model(input_layer: Layer, regression_layer: Layer, decoder_layer: Layer) -> Model:
    model = Model(
        input_layer,
        outputs=[regression_layer, decoder_layer]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main() -> None:
    data, labels = generate_data(2000)
    np.savetxt("data.csv", np.append(data, labels, axis=1), delimiter=";")

    layer_factory = LayerFactory()

    input_layer = Input(shape=(6,))
    encoder_layer = layer_factory.create("encoder")(input_layer)
    decoder_layer = layer_factory.create("decoder")(encoder_layer)
    regression_layer = layer_factory.create("regression")(encoder_layer)

    model = create_model(input_layer, regression_layer, decoder_layer)

    history = model.fit(data, [labels, data], epochs=50, validation_split=0.2).history

    encoder_model = Model(
        input_layer,
        encoder_layer
    )

    decoder_model = Model(
        input_layer,
        decoder_layer
    )

    regression_model = Model(
        input_layer,
        regression_layer
    )

    encoder_model.save("encoder_model.h5")
    decoder_model.save("decoder_model.h5")
    regression_model.save("regression_model.h5")

    encode_predicted_data = encoder_model.predict(data)
    decode_predicted_data = decoder_model.predict(data)
    regression_predicted_data = regression_model.predict(data)

    mean = data.mean(axis=0)
    std = data.std(axis=0)

    decode_predicted_data = decode_predicted_data * std + mean

    np.savetxt("encoded_data.csv", encode_predicted_data, delimiter=";")
    np.savetxt("decoded_data.csv", decode_predicted_data, delimiter=";")
    np.savetxt("regression.csv", np.append(labels, regression_predicted_data, axis=1), delimiter=";")


if __name__ == "__main__":
    main()
