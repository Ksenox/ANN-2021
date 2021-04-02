import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import optimizers
from numpy import asarray

res = dict()
los = dict()


def get_image(filename):
    image = Image.open(filename).convert('L')
    image = image.resize((28, 28))
    image = asarray(image) / 255.0
    output = np.expand_dims(image, axis=0)
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


def train_model(optimizer):
    optimizer_config = optimizer.get_config()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=5, batch_size=128,
                        validation_data=(test_images, test_labels))

    loss, acc = model.evaluate(test_images, test_labels)
    print('test_acc:', acc)

    plt.title('Training and validation accuracy')
    plt.plot(history.history['accuracy'], 'b', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'g', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #plt.savefig("%s_%s_%s_acc.png" % (optimizer_config["name"], optimizer_config["learning_rate"], acc), format='png')
    plt.clf()

    plt.title('Training and validation loss')
    plt.plot(history.history['loss'], 'b', label='Validation accuracy')
    plt.plot(history.history['val_loss'], 'g', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    #plt.savefig("%s_%s_%s_loss.png" % (optimizer_config["name"], optimizer_config["learning_rate"], acc), format='png')
    plt.clf()

    res["%s %s" % (optimizer_config["name"], optimizer_config["learning_rate"])] = acc
    los["%s %s" % (optimizer_config["name"], optimizer_config["learning_rate"])] = loss


for learning_rate in [0.001, 0.01]:
    train_model(optimizers.Adagrad(learning_rate=learning_rate))
    train_model(optimizers.Adam(learning_rate=learning_rate))
    train_model(optimizers.RMSprop(learning_rate=learning_rate))
    train_model(optimizers.SGD(learning_rate=learning_rate))


model.compile(optimizer=optimizers.Adagrad(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

print("accuracy:\n", res)
print("loss:\n", los)

image = get_image("6.png")
print(model.predict_classes(image))
