
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from keras.utils import to_categorical
from keras_preprocessing.image import load_img, img_to_array
from numpy import argmax
import numpy as np
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Ftrl

TEST_SIZE = 5


def load(f):
    img = Image.open(f)
    img = img.convert('L')
    img = img.resize((28, 28))
    return 1 - np.array(img) / 255.0


def test():
    for i in range(1, TEST_SIZE + 1):
        img = load('test/test_' + str(i) + '.png')
        digit = model.predict(np.array([img]))

        print("\nThe number in the picture is  " + str(i))
        print(digit[0])
        print("Predicted value: " + str(argmax(digit[0])))


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

opt = Nadam(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

test()

