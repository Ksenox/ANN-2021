from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def build_model():
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = build_model()

    model.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print(test_acc)

    model.save("model")


def load_image(path):
    img = load_img(path, color_mode="grayscale", target_size=(28, 28))

    vec = img_to_array(img)
    vec -= 255
    vec = vec / -255.0

    return vec


def run_model():
    model = load_model("model")
    img = load_image("./3.png")

    r = model.predict(np.asarray([img]))
    print(r)
    print(np.argmax(r, 1)[0])


train_model()
#run_model()