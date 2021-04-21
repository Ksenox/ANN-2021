from var1 import gen_data
import numpy as np
from tensorflow.keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from sklearn.preprocessing import LabelEncoder

data, label = gen_data(1000)

length = len(label)
test_data_length = length // 5

encoder = LabelEncoder()
encoder.fit(label)
labels = encoder.transform(label)

temp = list(zip(data, labels))
np.random.shuffle(temp)
data, labels = zip(*temp)
data = np.asarray(data).reshape(length, 50, 50, 1)
labels = np.asarray(labels).flatten()

train_data = data[test_data_length:length]
train_labels = labels[test_data_length:length]
test_data = data[:test_data_length]
test_labels = labels[:test_data_length]

model = Sequential()
model.add(Input(shape=(50, 50, 1)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=20, batch_size=10,
              validation_split=0.2)

evaluate = model.evaluate(test_data, test_labels)
