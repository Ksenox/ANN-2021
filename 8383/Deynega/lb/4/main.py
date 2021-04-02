from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
import tensorflow as tf
from PIL import Image
import numpy as np


def get_img(path):
    image_file = Image.open(path)
    image_file = image_file.convert('L')
    resized_image = image_file.resize((28, 28))
    resized_image.save("result.png", "PNG")
    return np.array(resized_image) / 255.0


def interface():
    print("Type path to 28x28 (recomended) .png file:")
    path = input()
    predict(load_model(), get_img(path))

def create_model():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    # opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,epsilon=None, decay=0.0, amsgrad=True, clipnorm=1., clipvalue=0.5)
    opt = tf.keras.optimizers.SGD(lr=0.9, momentum=0.0, decay=0.0, nesterov=False, clipnorm=1.0, clipvalue=0.5)
    # opt = tf.keras.optimizers.Adagrad(lr=0.18)
    # opt = tf.keras.optimizers.Nadam(learning_rate=0.003, beta_1=0.9, beta_2=0.999)
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")



def predict(model, img):
    # mnist = tf.keras.datasets.mnist
    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    #
    # test_images = test_images / 255.0
    # test_labels = to_categorical(test_labels)
    #
    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    # print('test_acc:', test_acc)
    pred = model.predict(np.array([img]))
    print(np.argmax(pred))

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    optim = tf.keras.optimizers.SGD(lr=0.9, momentum=0.0, decay=0.0, nesterov=False, clipnorm=1.0, clipvalue=0.5)
    loaded_model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model


interface()
