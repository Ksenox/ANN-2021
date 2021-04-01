import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import optimizers
from numpy import asarray

result = dict()


def get_image(filename):
    img = Image.open(filename).convert('L')
    img = img.resize((28, 28))
    img = asarray(img) / 255
    output = np.expand_dims(img, axis=0)
    return output


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))


def train_model(optimizer, epochs):
    optimizer_config = optimizer.get_config()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=128,
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)

    plt.title('Training and test accuracy')
    plt.plot(history.history['acc'], 'm', label='train')
    plt.plot(history.history['val_acc'], 'b', label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.clf()

    plt.title('Training and test loss')
    plt.plot(history.history['loss'], 'm', label='train')
    plt.plot(history.history['val_loss'], 'b', label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    result["%s %s" % (optimizer_config["name"], optimizer_config["learning_rate"])] = test_acc


for learning_rate in [0.001, 0.01]:
    train_model(optimizers.Adagrad(learning_rate=learning_rate), 5)
    train_model(optimizers.Adam(learning_rate=learning_rate), 5)
    train_model(optimizers.RMSprop(learning_rate=learning_rate), 5)
    train_model(optimizers.SGD(learning_rate=learning_rate), 5)

for res in result:
    print(res, ':', result[res])

model.compile(optimizer=optimizers.Adagrad(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

image = get_image("test.png")
print(image)
print(model.predict_classes(image))
