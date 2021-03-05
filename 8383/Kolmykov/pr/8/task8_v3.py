import task8_generator as generator
import numpy as np
import random
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.callbacks import Callback
import matplotlib.pyplot as plt


def draw_hist(accuracy, val_accuracy, number):
    epochs = range(1, len(accuracy) + 1)
    plt.clf()
    plt.plot(epochs, accuracy, 'bo', label='Training acc')
    plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig("hist_" + str(number) + ".png")

def shuffle_data(data, label):
    tmp_data = []
    for i in range(len(data)):
        tmp_data.append([data[i], label[i]])
    random.shuffle(tmp_data)
    res_data = []
    res_label = []
    for i in range(len(data)):
        res_data.append(tmp_data[i][0])
        res_label.append(tmp_data[i][1])
    return np.asarray(res_data), np.asarray(res_label)


def split_data(data, label):
    size = len(data)
    test_size = size // 5
    train_size = size - test_size
    val_size = train_size // 5
    train_size -= val_size
    return (data[0:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]), \
           (label[0:train_size], label[train_size:train_size + val_size], label[train_size + val_size:])



x, y = generator.gen_data()
print(x.shape)
y = np.asarray([[0.] if item == 'Empty' else [1.] for item in y])
x, y = shuffle_data(x, y)
(train_x, val_x, test_x), (train_y, val_y, test_y) = split_data(x, y)

batch_size = 32
num_epochs = 10

kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
hidden_size = 512


inp = Input(shape=(50, 50, 1))

conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu', data_format='channels_last')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu', data_format='channels_last')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu', data_format='channels_last')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu', data_format='channels_last')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_1)(hidden)
out = Dense(1, activation='sigmoid')(drop_3)

model = Model(inputs=[inp], outputs=[out])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs_for_callback = np.zeros(num_epochs)
print("Введите индексы эпох для вывода гистограммы через пробел:")
input_arr = input().split()
for ind in input_arr:
    if int(ind) < num_epochs:
        epochs_for_callback[int(ind)] = 1
accuracy_list = []
val_accuracy_list = []


class HistogramCallBack(Callback):
    def on_epoch_end(self, epoch, logs=None):
        accuracy_list.append(logs['accuracy'])
        val_accuracy_list.append(logs['val_accuracy'])
        if epochs_for_callback[epoch] == 1:
            draw_hist(accuracy_list, val_accuracy_list, epoch)


model.fit(train_x, train_y,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_data=(val_x, val_y),
          callbacks=[HistogramCallBack()])
model.evaluate(test_x, test_y, verbose=True)
