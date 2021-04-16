import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import MaxPooling2D, Convolution2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def plot_loss(loss, v_loss):
    plt.figure(1, figsize=(8, 5))
    plt.plot(loss, 'b', label='train')
    plt.plot(v_loss, 'r', label='validation')
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()


def plot_acc(acc, val_acc):
    plt.plot(acc, 'b', label='train')
    plt.plot(val_acc, 'r', label='validation')
    plt.title('acc')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

num_train, depth, height, width = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_train) # Normalise data to [0, 1] range

Y_train = to_categorical(y_train, num_classes) # One-hot encode the labels
Y_test = to_categorical(y_test, num_classes) # One-hot encode the labels

inp = Input(shape=(depth, height, width))  # N.B. depth goes first in Kerass
conv_1 = Convolution2D(32, (3, 3), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(32, 3, 3, padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)
drop_1 = Dropout(0.25)(pool_1)
conv_3 = Convolution2D(64, 3, 3, padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(64, 3, 3, padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=1)(conv_4)
drop_2 = Dropout(0.25)(pool_2)
flat = Flatten()(drop_2)
hidden = Dense(512, activation='relu')(flat)
drop_3 = Dropout(0.5)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)
model = Model(inputs=[inp], outputs=[out])

model.compile(Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])

H = model.fit(X_train, Y_train, batch_size=100, epochs=15, verbose=1, validation_split=.1)

l, acc = model.evaluate(X_test, Y_test)
print('test data', acc)

plot_loss(H.history['loss'], H.history['val_loss'])
plot_acc(H.history['acc'], H.history['val_acc'])