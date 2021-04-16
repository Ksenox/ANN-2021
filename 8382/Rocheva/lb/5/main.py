from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
num_epochs = 60
kernel_size = 5  # размер ядра
pool_size = 2  # размер подвыборки в слоях подвыборки
conv_depth_1 = 32  # количество ядер
conv_depth_2 = 64
drop_prob_1 = 0.25  # dropout
drop_prob_2 = 0.5
hidden_size = 512  # кол-во нейронов в полносвязном слое MLP

(X_train, y_train), (X_test, y_test) = cifar10.load_data()  # fetch CIFAR-10 data

num_train, depth, height, width = X_train.shape  # there are 50000 training examples in CIFAR-10
num_test = X_test.shape[0]  # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0]  # there are 10 image classes

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train)  # Normalise data to [0, 1] range
X_test /= np.max(X_train)  # Normalise data to [0, 1] range

Y_train = to_categorical(y_train, num_classes)  # One-hot encode the labels
Y_test = to_categorical(y_test, num_classes)  # One-hot encode the labels

inp = Input(shape=(depth, height, width))  # N.B. depth goes first in Keras

conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)

conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

# Now flatten to 1D, apply Dense -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)

hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1)
model.evaluate(X_test, Y_test, verbose=1)
