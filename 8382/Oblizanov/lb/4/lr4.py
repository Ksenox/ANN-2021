import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from PIL import Image
import numpy as np


def load_image(path):
    img = Image.open(path).convert('L')
    img_data = np.array(img)
    if img_data.size != 784 or len(img_data) != 28:
        img = img.resize((28, 28), Image.ANTIALIAS)
        img_data = np.array(img)
    img_data = img_data / 255.0
    return np.expand_dims(img_data, axis=0)


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


def create_model(opt='adam'):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, batch_size=256, verbose=0)
    return model, history


opts = {
    'RMSProp,lr=.001': optimizers.RMSprop(learning_rate=0.001),
    'RMSProp,lr=.01': optimizers.RMSprop(learning_rate=0.01),
    'RMSProp,lr=.1': optimizers.RMSprop(learning_rate=0.1),
    'Adam,lr=.001,e=1e-7': optimizers.RMSprop(learning_rate=0.001),
    'Adam,lr=.01,e=1e-7': optimizers.Adam(learning_rate=0.01),
    'Adam,lr=.1,e=1e-7': optimizers.Adam(learning_rate=0.1),
    'Adam,lr=.001,e=1e-4': optimizers.Adam(learning_rate=0.001, epsilon=0.0001),
    'Adamax,lr=.001': optimizers.Adamax(learning_rate=0.001),
    'Adamax,lr=.01': optimizers.Adamax(learning_rate=0.01),
    'Adamax,lr=.1': optimizers.Adamax(learning_rate=0.1),
    'SGD,lr=.001': optimizers.Adamax(learning_rate=0.001),
    'SGD,lr=.01': optimizers.Adamax(learning_rate=0.01),
    'SGD,lr=.1': optimizers.Adamax(learning_rate=0.1)
}


def test_optimizers(opts):
    stats = {}
    result = {}
    epochs = range(1, 11)
    res_model = None
    max_acc = 0
    best_opt = None
    for p in opts.keys():
        print(p)
        model, history = create_model(opts[p])
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        if test_acc > max_acc:
            res_model = model
            max_acc = test_acc
            best_opt = p
        print("Accuracy:", test_acc)
        print("---------------------")
        stats[p] = [history.history["accuracy"], history.history["loss"]]
        result[p] = test_acc
    for p in stats.keys():
        plt.plot(epochs, stats[p][0], label=p)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    for p in stats.keys():
        plt.plot(epochs, stats[p][1], label=p)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.bar(stats.keys(), result.values())
    plt.title("Accuracy")
    plt.show()
    print(result.values())
    return res_model, best_opt


model, best_opt = test_optimizers(opts)
print("Testing best model with optimizer", best_opt, "on custom images:")
image = load_image('test1.png')
print('Loaded image of "1":', model.predict_classes(image))
image = load_image('test2.png')
print('Loaded image of "2":', model.predict_classes(image))
image = load_image('test4.png')
print('Loaded image of "4":', model.predict_classes(image))
image = load_image('test8.png')
print('Loaded image of "8":', model.predict_classes(image))
