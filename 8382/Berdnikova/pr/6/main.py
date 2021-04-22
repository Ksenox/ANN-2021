from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt
from var1 import gen_data

x, y = gen_data(1000)
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
data = list(zip(x, y))
np.random.shuffle(data)
x, y = zip(*data)
x = np.asarray(x).reshape(sz, 50, 50, 1)
y = np.asarray(y)
x_test = x[:len(y) // 5]
x_train = x[len(y) // 5:len(y)]
y_test = y[:len(y) // 5]
y_train = y[len(y) // 5:len(y)]
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(50, 50, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=11, batch_size=10, validation_split = 0.1)
model.evaluate(x_test, y_test)