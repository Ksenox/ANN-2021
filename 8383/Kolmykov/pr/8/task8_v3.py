from sklearn.preprocessing import LabelEncoder

import task8_generator as generator
import numpy as np
import random
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils


def draw_hist(correctPredictions, wrongPredictions, epoch):
    data = {'Correct': correctPredictions,
            'Wrong': wrongPredictions}

    df = pd.DataFrame(data)
    df.plot(kind='bar', stacked=True)
    plt.savefig("hist_" + str(epoch + 1) + ".png")
    plt.show()


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
# print(x.shape)
# y = np.asarray([[0.] if item == 'Empty' else [1.] for item in y])
x, y = shuffle_data(x, y)
# (train_x, val_x, test_x), (train_y, val_y, test_y) = split_data(x, y)
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y, 2)

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
out = Dense(2, activation='softmax')(drop_3)

model = Model(inputs=[inp], outputs=[out])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs_for_callback = np.zeros(num_epochs)
print("Введите индексы эпох для вывода гистограммы через пробел:")
input_arr = input().split()
for ind in input_arr:
    if int(ind) < num_epochs:
        epochs_for_callback[int(ind) - 1] = 1
accuracy_list = []
val_accuracy_list = []


class HistogramCallBack(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # accuracy_list.append(logs['accuracy'])
        # val_accuracy_list.append(logs['val_accuracy'])
        # if epochs_for_callback[epoch] == 1:
        #     draw_hist(accuracy_list, val_accuracy_list, epoch)

        if epochs_for_callback[epoch] != 0:

            correctPredictions = [0, 0]
            wrongPredictions = [0, 0]
            countsPredictions = [0, 0]
            predictions = model.predict(x)

            for i in range(len(predictions)):
                if (np.argmax(predictions[i]) == np.argmax(y[i])):
                    correctPredictions[np.argmax(y[i])] += 1
                else:
                    wrongPredictions[np.argmax(y[i])] += 1

                countsPredictions[np.argmax(y[i])] += 1

            correctPredictions = np.divide(correctPredictions, countsPredictions)
            wrongPredictions = np.divide(wrongPredictions, countsPredictions)
            draw_hist(correctPredictions, wrongPredictions, epoch)


model.fit(x, y,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1,
          callbacks=[HistogramCallBack()])

x_test, y_test = generator.gen_data(size=1000, img_size=50)
encoder = LabelEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
y_test = np_utils.to_categorical(y_test, 2)
model.evaluate(x_test, y_test, verbose=True)
