import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing.image import *
import numpy as np


def load_image(path):
    img = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(28, 28))
    return 1 - np.reshape(img_to_array(img) / 255, (28, 28))


def user_image():
    for img in range(10):
        path = "num/" + str(img) + ".png"
        loaded_image = load_image(path)
        res = model.predict(np.reshape(loaded_image, (-1, 28, 28)))
        print("На картинке -", str(img), " Предсказание -", np.argmax(res))


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adadelta(learning_rate=1.5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('test_acc:', test_acc)

user_image()
