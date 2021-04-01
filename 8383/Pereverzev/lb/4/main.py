from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from PIL import Image
import numpy as np
mnist = tf.keras.datasets.mnist


def get_img(path):
    image_file = Image.open(path)
    image_file = image_file.convert('L')
    resized_image = image_file.resize((28, 28))
    return np.array(resized_image) / 255.0


def drawHistory(history):
    loss = history['loss']
    acc = history['accuracy']
    epochs = range(1, len(loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle('Loss and accuracy')
    ax1.plot(epochs, loss, color="green", label='Loss')
    ax1.legend()
    ax2.plot(epochs, acc, color="green", label='Accuracy')
    ax2.legend()
    plt.show()


def func(path):
    # plt.imshow(train_images[idx], cmap=plt.cm.binary)
    # plt.show()
    # print(train_labels[idx])
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # opt = RMSprop(learning_rate=0.1, rho=0.9, centered=True)
    # opt = RMSprop(learning_rate=0.01, rho=0.9, centered=True)
    # opt = RMSprop(learning_rate=0.001, rho=0.9, centered=True)
    # opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    # opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    # opt = Adam(learning_rate=0.0001, beta_1=0.99, beta_2=0.999)
    # opt = Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    # opt = Nadam(learning_rate=0.001, beta_1=0.99, beta_2=0.999)
    opt = Nadam(learning_rate=0.001, beta_1=0.999, beta_2=0.999)

    model.compile(optimizer="nadam", loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels,
                        epochs=5, batch_size=128, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    # drawHistory(history.history)
    ress = model.predict(np.array([get_img(path)]))
    print(np.argmax(ress))


func("./img/1.png")
# func("./img/2.png")
# func("./img/3.png")
# func("./img/4.png")
# func("./img/5.png")
# func("./img/6.png")
# func("./img/7.png")
# func("./img/8.png")
# func("./img/9.png")
# func("./img/0.png")
