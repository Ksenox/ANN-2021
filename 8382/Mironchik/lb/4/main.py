import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing.image import *
import numpy as np


def load_image(path):
    img = tf.keras.preprocessing.image.load_img(
        path,
        color_mode='grayscale',
        target_size=(28, 28),
    )
    return 1 - np.reshape(img_to_array(img) / 255, (28, 28))


def show_image(img):
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


optimizersSGD = [
    SGD(),  # 0.91
    SGD(learning_rate=0.001),  # 0.82
    SGD(learning_rate=0.1),  # 0.96
    SGD(momentum=0.1),  # 0.91
    SGD(momentum=0.6),  # 0.93
    SGD(momentum=0.9),  # 0.95
    SGD(nesterov=True),  # 0.91
    SGD(learning_rate=0.1, momentum=0.9),  # 0.977
]

optimizersRMSprop = [
    RMSprop(),  # 0.975
    RMSprop(learning_rate=0.0001),  # 0.94
    RMSprop(learning_rate=0.01),  # 0.97
    RMSprop(learning_rate=0.1),  # 0.89
    RMSprop(rho=0.7),  # 0.977
    RMSprop(rho=0.5),  # 0.975
]

optimizersAdagrad = [
    Adagrad(),  # 0.87
    Adagrad(0.1),  # 0.973
    Adagrad(0.001),  # 0.877
]

optimizersAdadelta = [
    Adadelta(),  # 0.6
    Adadelta(learning_rate=1.5),  # 0.978
    Adadelta(learning_rate=0.8),  # 0.974
    Adadelta(learning_rate=0.6),  # 0.972
    Adadelta(rho=0.99),  # 0.72
    Adadelta(rho=0.8),  # 0.41
]

optimizersAdam = [
    Adam(),  # 0.977
    Adam(learning_rate=0.01),  # 0.966
    Adam(learning_rate=0.1),  # 0.76
    Adam(learning_rate=0.0001),  # 0.94
    Adam(amsgrad=True),  # 0.977
]

optimizersAdamax = [
    Adamax(),  # 0.967
    Adamax(learning_rate=0.001),  # 0.964
    Adamax(learning_rate=0.01),  # 0.977
    Adamax(learning_rate=0.1),  # 0.956
    Adamax(learning_rate=0.0002),  # 0.93
]

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adadelta(learning_rate=1.5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=0)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('test_acc:', test_acc)

images = [
    "0.png",
    "1.png",
    "2.png",
    "3.png",
    "4.png",
    "5.png",
    "6.png",
    "7.png",
    "8.png",
    "9.png",
]

for img in images:
    path = "img/"+img
    loaded_image = load_image(path)
    show_image(loaded_image)
    res = model.predict(np.reshape(loaded_image, (-1, 28, 28)))
    print(path, "is probably", np.argmax(res))
