import sys

from typing import Tuple, List

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import models
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.utils.np_utils import to_categorical


def load_data() -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return (train_images, train_labels), (test_images, test_labels)


def create_model() -> models.Model:
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_model(model: models.Model, train_data: np.array, train_labels: np.array, batch_size: int, epochs: int) -> History:
    return model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)


def load_image_to_array(path: str) -> np.array:
    image = load_img(path, color_mode="grayscale", target_size=(28, 28))
    input_arr = img_to_array(image)

    input_arr -= 255
    input_arr /= -255

    ret_array = np.array([input_arr])
    return ret_array


def main(images: List[str]):
    (train_data, train_labels), (test_data, test_labels) = load_data()

    model = create_model()
    train_model(model, train_data, train_labels, 128, 5)

    results = model.evaluate(test_data, test_labels)
    print(results)

    for image_path in images:
        image = load_image_to_array(image_path)
        print(f"Prediction for {image_path}: {np.argmax(model.predict(image))}")


if __name__ == "__main__":
    main(sys.argv[1:])
