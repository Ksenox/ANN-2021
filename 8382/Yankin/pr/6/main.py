from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
import numpy as np
from matplotlib import pyplot as plt
from var5 import gen_data


[data, labels] = gen_data(1000)
[data, labels] = shuffle(data, labels)

data /= np.max(data)

encoder = LabelEncoder()
encoder.fit(labels.ravel())
labels = encoder.transform(labels.ravel())


data_len, height, width = data.shape

validation_ratio = 0.2
test_ratio = 0.2

train_data = data[: int(data_len*(1 - validation_ratio - test_ratio))]
train_labels = labels[: int(data_len*(1 - validation_ratio - test_ratio))]

validation_data = data[int(data_len*(1 - validation_ratio - test_ratio)): int(data_len*(1 - test_ratio))]
validation_labels = labels[int(data_len*(1 - validation_ratio - test_ratio)): int(data_len*(1 - test_ratio))]

test_data = data[int(data_len*(1 - test_ratio)):]
test_labels = labels[int(data_len*(1 - test_ratio)):]


train_data = train_data.reshape(train_data.shape[0], width, height, 1)
validation_data = validation_data.reshape(validation_data.shape[0], width, height, 1)
test_data = test_data.reshape(test_data.shape[0], width, height, 1)


kernel_size = 3
pool_size = 2
conv_depth_1 = 8
conv_depth_2 = 16
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 128
batch_size = 32
num_epochs = 15


model = Sequential()

model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding="same", activation='relu',
                        input_shape=(width, height, 1)))
model.add(Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_1))

model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding="same", activation='relu'))
model.add(Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_1))

model.add(Flatten())
model.add(Dense(hidden_size, activation='relu'))
model.add(Dropout(drop_prob_2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=num_epochs, verbose=1,
          validation_data=(validation_data, validation_labels))
history = history.history

model.evaluate(test_data, test_labels)


plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()
