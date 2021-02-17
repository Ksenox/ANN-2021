import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from PIL import Image
import numpy as np


class SizeException(Exception): pass


def interface():
    while(True):
        print('Input path to the file (or "stop"):')
        path = input()
        if path == 'stop':
            break
        try:
            img = get_image(path)
            predict(img)
        except PermissionError:
            print("Error: Permission denied")
            continue
        except FileNotFoundError:
            print("Error: File doesn't exists")
            continue
        except SizeException:
            print("Error: Size is not 28x28")
            continue


def predict(img):
    pred = model.predict(img)
    print(np.argmax(pred))


def get_image(path):
    im = Image.open(path)
    arr = np.asarray(im)

    if len(arr) != 28 | len(arr[0]) != 28:
        print(arr)
        raise SizeException

    if arr[0][0].ndim != 0:
         arr = fix_arr(arr)

    arr = arr / 255.0
    t = np.array([arr])
    return t


def fix_arr(arr):
    res = np.zeros((len(arr), len(arr[0])), dtype=int)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            res[i][j] = (arr[i][j][0] + arr[i][j][1] + arr[i][j][2])//3
    return res


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()
# print(train_images[0])
# print(np.shape(train_images[0]))

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
# opt = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=False)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

interface()
