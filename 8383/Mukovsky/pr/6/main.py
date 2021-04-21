import tensorflow as tf
device_name = tf.test.gpu_device_name()

import sklearn.utils
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn.preprocessing import LabelEncoder
from matplotlib import gridspec


def gen_circle(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    r = np.random.randint(size // 10, size // 3)
    for i in range(0, size):
        for j in range(0, size):
            if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                img[i, j] = 1
    return img


def gen_empty_circle(size=50):
    img = np.zeros([size, size])
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    r = np.random.randint(size // 10, size // 3)
    dr = np.random.randint(1, 10) + r
    for i in range(0, size):
        for j in range(0, size):
            if r ** 2 <= (i - x) ** 2 + (j - y) ** 2 <= dr ** 2:
                img[i, j] = 1
    return img


def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1

    label_c1 = np.full([c1, 1], 'Empty')
    data_c1 = np.array([gen_empty_circle(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Not Empty')
    data_c2 = np.array([gen_circle(img_size) for i in range(c2)])

    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))

    return data, label


def prepare_data(X, Y):
    X, Y = sklearn.utils.shuffle(X, Y)
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    val_split = 0.1
    test_num = round((1 - val_split) * len(X))
    val_num = round(test_num * (1 - val_split))
    train_data, train_labels = X[:val_num], Y[:val_num]
    val_data, val_labels = X[val_num:test_num], Y[val_num:test_num]
    test_data, test_labels = X[test_num:], Y[test_num:]
    train_data = np.expand_dims(train_data, axis=3)
    val_data = np.expand_dims(val_data, axis=3)
    test_data = np.expand_dims(test_data, axis=3)
    return (train_data, val_data, test_data), (train_labels, val_labels, test_labels)


data, labels = gen_data()
(train_data, val_data, test_data), (train_labels,
                                    val_labels, test_labels) = prepare_data(data, labels)

nums, width, height = data.shape

batch_size = 15
num_epochs = 15
kernel_size = 3
pool_size = 2
conv_depth_1 = 16
conv_depth_2 = 32
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 64

inp = Input(shape=(height, width, 1))

conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                       padding='same', activation='relu')(inp)

pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)

drop_1 = Dropout(drop_prob_1)(pool_1)

conv_2 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                       padding='same', activation='relu')(drop_1)

pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                       padding='same', activation='relu')(pool_2)

drop_1 = Dropout(drop_prob_1)(conv_3)

flat = Flatten()(drop_1)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_1 = Dropout(drop_prob_2)(hidden)
out = Dense(1, activation='sigmoid')(drop_1)

model = Model(inputs=[inp], outputs=[out])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(val_data, val_labels))

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 3])
plt.subplot(gs[0])
plt.plot(epochs, loss, 'b--', label='Training loss')
plt.plot(epochs, val_loss, 'r--', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(gs[1])
plt.plot(epochs, acc, 'b--', label='Training acc')
plt.plot(epochs, val_acc, 'r--', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.evaluate(test_data, test_labels, verbose=1)
