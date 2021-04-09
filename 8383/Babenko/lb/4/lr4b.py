import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(X_test[84], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_test[142], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_test[139], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_test[35], cmap=plt.get_cmap('gray'))
plt.show()

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def baseline_model(optimizers_list, labels):
    acc_list = []
    history_loss_list = []
    history_acc_list = []

    for opt in optimizers_list:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit(X_train, y_train, epochs=5, batch_size=128)
        history_loss_list.append(hist.history['loss'])
        history_acc_list.append(hist.history['accuracy'])
        test_loss, test_acc = model.evaluate(X_test, y_test)
        acc_list.append(test_acc)

    print("-------------------------------")
    print("Точность = " + str(np.round(acc_list, 2)))
    x = range(1, 6)

    plt.subplot(211)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for loss in history_loss_list:
        plt.plot(x, loss)
    plt.legend(labels)
    plt.grid()

    plt.subplot(212)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for acc in history_acc_list:
        plt.plot(x, acc)
    plt.legend(labels)
    plt.grid()
    plt.show()

    return model


optimizers_list = []

optimizers_list.append(optimizers.SGD())
optimizers_list.append(optimizers.SGD(learning_rate=0.1, momentum=0.0))
optimizers_list.append(optimizers.SGD(learning_rate=0.1, momentum=0.8))
optimizers_list.append(optimizers.SGD(learning_rate=0.01, momentum=0.8))
baseline_model(optimizers_list, (
"SGD(default)", "SGD(learning_rate=0.1, momentum=0.0)", "SGD(learning_rate=0.1, momentum=0.8)",
"SGD(learning_rate=0.01, momentum=0.8)"))

optimizers_list.append(optimizers.RMSprop())  # default learning_rate=0.001, rho=0.9
optimizers_list.append(optimizers.RMSprop(learning_rate=0.01, rho=0.9))
optimizers_list.append(optimizers.RMSprop(learning_rate=0.01, rho=0.5))
optimizers_list.append(optimizers.RMSprop(learning_rate=0.001, rho=0.5))
baseline_model(optimizers_list, (
"RMSprop(default)", "RMSprop(learning_rate=0.01, rho = 0.9)", "RMSprop(learning_rate=0.01, rho = 0.5)",
"RMSprop(learning_rate=0.001, rho = 0.5)"))

optimizers_list.append(optimizers.Adam())  # default (lr=0.001, beta_1=0.9, beta_2=0.999)
optimizers_list.append(optimizers.Adam(learning_rate=0.01, beta_1=0.9))
optimizers_list.append(optimizers.Adam(learning_rate=0.01, beta_1=0.1))
optimizers_list.append(optimizers.Adam(learning_rate=0.1, beta_1=0.9))
model = baseline_model(optimizers_list, ("Adam(default)", "Adam(learning_rate=0.01, beta_1=0.9)",
                                 "Adam(learning_rate=0.01, beta_1=0.1)",
                                 "Adam(learning_rate=0.1, beta_1=0.9)",
                                 ))


def read_and_predict(path, model):
    image = Image.open(path).convert('L')
    data = asarray(image)
    data = data.reshape((1, 28, 28))
    Y = model.predict_classes(data)
    print("prediction for file " + path + " -> " "[ " + np.array2string(Y[0]) + " ]")
    return data

read_and_predict("n_2.png", model)
read_and_predict("n_3.png", model)
read_and_predict("n_4.png", model)
read_and_predict("n_8.png", model)
