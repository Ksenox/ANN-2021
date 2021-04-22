from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from var7 import gen_data

data, labels = gen_data(5000)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train.reshape(x_train.shape[0], 50, 50, 1)
x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)

encoder = LabelEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
y_test = to_categorical(y_test)
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_train = to_categorical(y_train)

model = Sequential()
model.add(Conv2D(16, kernel_size=(7, 7), activation='relu', input_shape=(50, 50, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
H = model.fit(x_train, y_train, epochs=12, batch_size=20, validation_split = 0.1)
model.evaluate(x_test, y_test)