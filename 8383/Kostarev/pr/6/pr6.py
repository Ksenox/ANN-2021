import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import var4

batch_size = 32
num_epochs = 10
kernel_size = 5
pool_size = 2
conv_depth_1 = 32
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

x_data, y_data = var4.gen_data()

label_encoder = LabelEncoder()
label_encoder.fit(np.unique(y_data))
y_encoded = label_encoder.transform(y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_encoded, test_size=0.33)

inp = Input(shape=(*x_data.shape[1:], 1))
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
flat = Flatten()(drop_1)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_2 = Dropout(drop_prob_2)(hidden)
out = Dense(1, activation='sigmoid')(drop_2)

model = Model(inputs=inp, outputs=out)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1)
evaluate = model.evaluate(x_test, y_test, verbose=1)
