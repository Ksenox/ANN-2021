import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import MaxPooling2D, Dense, Dropout, Flatten
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D

from MyCallback import MyCallback


def gen_rect(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    w = np.random.randint(size // 10, size // 2)
    h = np.random.randint(size // 10, size // 2)
    img[x:x + w, y:y + h] = 1
    return img


def gen_circle(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    r = np.random.randint(size // 10, size // 3)
    for i in range(0, size):
        for j in range(0, size):
            if (i-x)**2 + (j-y)**2 <= r**2:
                img[i, j] = 1
    return img


def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1
    label_c1 = np.full([c1, 1], 'Square')
    data_c1 = np.array([gen_rect(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Circle')
    data_c2 = np.array([gen_circle(img_size) for i in range(c2)])
    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))
    return data, label


img_size = 40
size = 1000

data, labels = gen_data(size, img_size)

data_len, height, width = data.shape

data /= np.max(data)
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.fit_transform(labels.ravel())

rand_index = np.random.permutation(len(labels))
data, labels = data[rand_index], labels[rand_index]

val_size = 0.2

train_data, val_data = data[: int(data_len * (1 - val_size))], data[int(data_len*(1 - val_size)):]
train_labels, val_labels = labels[: int(data_len * (1 - val_size))],  labels[int(data_len*(1 - val_size)):]

train_data = train_data.reshape(train_data.shape[0], width, height, 1)
val_data = val_data.reshape(val_data.shape[0], width, height, 1)

batch_size = 32
num_epochs = 15
kernel_size = 3
pool_size = 2
conv_depth_1 = 16
conv_depth_2 = 32
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

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
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

date = str(datetime.date.today())
prefix = input("Введите префикс: ")
model_save_callback = MyCallback(prefix)

history = model.fit(train_data, train_labels,
              batch_size=batch_size, epochs=num_epochs, callbacks=[model_save_callback],
              verbose=1, validation_split=0.2)

history = history.history

model.evaluate(val_data, val_labels, verbose=2)


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