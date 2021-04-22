import numpy as np
import tensorflow.keras as keras
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from var6 import gen_data

def getData():
    data, labels = gen_data(size=1000)
    data, labels = shuffle(data, labels)
    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, labels, test_size=0.2, random_state=11)
    dataTrain = dataTrain.reshape(dataTrain.shape[0], 50, 50, 1)
    dataTest = dataTest.reshape(dataTest.shape[0], 50, 50, 1)

    encoder = LabelEncoder()
    encoder.fit(labelTrain)
    labelTrain = encoder.transform(labelTrain)
    labelTrain = to_categorical(labelTrain)

    encoder.fit(labelTest)
    labelTest = encoder.transform(labelTest)
    labelTest = to_categorical(labelTest)
    return dataTrain, labelTrain, dataTest, labelTest

def createModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding="same", activation='relu', input_shape=(50, 50, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, kernel_size=3,padding="same", strides=1, activation='relu'))
    model.add(Conv2D(64, kernel_size=3,padding="same", strides=1, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def drawPlots(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dataTrain, labelTrain, dataTest, labelTest = getData()
    model = createModel()
    history = model.fit(dataTrain, labelTrain, epochs=10, batch_size=10,validation_split=0.1)
    model.evaluate(dataTest, labelTest)
    drawPlots(history)