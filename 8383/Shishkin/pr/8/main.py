import sklearn.metrics
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.utils import np_utils
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

import var3


# var. 6
def breakdown_of_data(data, label):
    size = len(data)
    test_size = size // 5
    test_data = data[:test_size]
    test_label = label[:test_size]
    data = data[test_size:]
    label = label[test_size:]

    val_size = size // 5
    val_data = data[:val_size]
    val_label = label[:val_size]
    data = data[val_size:]
    label = label[val_size:]

    return (data[:], val_data, test_data), (label[:], val_label, test_label)


num_classes = 2
X, Y = var3.gen_data()  # rang(X) = 3, rang(Y) = 2
X, Y = shuffle(X, Y)  # т.к. данные не перемешаны
encoder = LabelEncoder()  # т.к. классы характеризуются строковой меткой
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
encoded_Y = np_utils.to_categorical(encoded_Y, num_classes)
(train_x, val_x, test_x), (train_y, val_y, test_y) = breakdown_of_data(X, encoded_Y)

batch_size = 30  # in each iteration, we consider 1 training example at once
num_epochs = 10  # we iterate 10 times over the entire training set
kernel_size = 4  # we will use 4x4 kernels throughout
pool_size = 4  # we will use 4x4 pooling throughout
conv_depth_1 = 32  # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64  # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
drop_prob_2 = 0.5  # dropout in the dense layer with probability 0.5
hidden_size = 512  # the dense layer will have 512 neurons

inp = Input(shape=(X.shape[1], X.shape[2], 1))  # N.B. depth goes first in Keras

# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

# Now flatten to 1D, apply Dense -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)  # To define a model, just specify its input and output layers

model.compile(loss='binary_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',  # using the Adam optimiser
              metrics=['accuracy'])  # reporting the accuracy

print("Enter interval:")
interval = int(input())
while interval <= 0:
    interval = int(input())

th = ['Epoch num', 'Worst accuracy num', 'Class', 'Accuracy', 'Loss']
td = []
columns = len(th)
table = PrettyTable(th)


class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        super(MyCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch == num_epochs - 1 or epoch % interval == 0:
            yPred = model.predict(train_x)
            yTrue = train_y
            minAccInd = np.argmin(1 - np.abs(yTrue - yPred))

            acc = logs.get("accuracy")
            loss = logs.get("loss")
            which_class = 1
            classification = train_y.flatten()[minAccInd]
            if classification == 1.0:
                which_class = 0
            td.append(epoch)
            td.append(minAccInd)
            td.append(which_class)
            td.append(round(acc, 3))
            td.append(round(loss, 3))


model.fit(train_x, train_y,  # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=2, validation_data=(val_x, val_y), callbacks=MyCallback())

evaluate_model = model.evaluate(test_x, test_y, verbose=2, callbacks=MyCallback())

while td:
    table.add_row(td[:columns])
    td = td[columns:]
print(table)
with open('table.txt', 'w') as w:
    w.write(str(table))
