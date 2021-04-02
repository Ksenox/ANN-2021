import tensorflow as tf
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from keras import optimizers


def loadImage(path):
    img = Image.open(path).convert('L')
    arr = np.array(img) / 255.0
    arr = np.array([arr])

    if arr.shape != (1, 28, 28): raise Exception("Размер изображения неверный")

    return arr


mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation= 'relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer= optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

try:

    if 'y' == input("Do you want to enter your file? (y/n) "):

        while(True):
            path = input("Enter path to file or 0 to exit: ")

            if path == '0':
                break

            predicted_results = model.predict(loadImage(path))
            print("I predict that is ", list(np.array(predicted_results[0])).index(max(predicted_results[0])))


except Exception as ex:
    print(ex)



