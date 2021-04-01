import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt


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

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


def dialog():
    while True:
        print('Введите адрес изображения')
        string = input()

        if string == 'exit':
            break

        try:
            image = load_image(string)
            res = model.predict(image)
            print(np.argmax(res))
        except FileNotFoundError:
            print('Файл не найден')
        except PIL.UnidentifiedImageError:
            print('Неверный формат файла')
        except ValueError:
            print('Неверный размер изображения')
        except Exception:
            print('Что-то не так')
        finally:
            print('')


def load_image(path):
    image = Image.open(path).convert('L')
    image = np.asarray(image) / 255.0
    if image.shape != (28, 28):
        raise ValueError
    return np.array([image])


dialog()
