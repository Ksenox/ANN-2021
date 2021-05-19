from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras import utils
import numpy as np

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import matplotlib.pyplot as plt

import var5

batch_size = 32
num_epochs = 200
kernel_size = 4
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

size = 200
img_size = 50

data, labels = var5.gen_data(size,img_size)
labels.reshape(labels.size)
num_classes = np.unique(labels).shape[0]
le.fit(np.unique(labels))
labels = le.transform(labels)
labels = utils.to_categorical(labels, num_classes) # One-hot encode the labels

inp = Input(shape=(img_size, img_size,1))

# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
layer = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
layer = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(layer)
# layer = MaxPooling2D(pool_size=(pool_size, pool_size))(layer)
# layer = Dropout(drop_prob_1)(layer)


flat = Flatten()(layer)

layer = Dense(hidden_size, activation='relu')(flat)

out = Dense(1, activation='relu')(layer)


model = Model(inputs=inp, outputs=out)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

H = model.fit(data,labels,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=1,
              validation_split=.1)

model.evaluate(data, labels, verbose=1)

import pandas



