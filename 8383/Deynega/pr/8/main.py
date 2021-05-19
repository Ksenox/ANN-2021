import var6
import numpy as np
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

table = []

class MinAccJournal(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

    def on_epoch_end(self, epoch, logs=None):
        min_acc = min(self.accuracy)
        min_acc_ind = self.accuracy.index(min_acc)
        min_loss = self.losses[min_acc_ind]
        obs_class = labels[min_acc_ind]
        self.losses = []
        self.accuracy = []
        if epoch % interval == 0 or epoch == num_epochs - 1:
            table.append([epoch, min_acc_ind, obs_class[0], min_acc, min_loss])

(data, labels) = var6.gen_data()

interval = int(input("Type output interval:\n"))
interval+=1

size, height, width = data.shape
batch_size = 1
num_epochs = 10
kernel_size = 4
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512
num_classes = data.shape[0]

data, labels = shuffle(data, labels)
data = data.astype('float32')
data /= np.max(data)
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
encoded_Y = np.reshape(encoded_Y, (size, 1))
encoded_Y = np_utils.to_categorical(encoded_Y, num_classes)


inp = Input(shape=(height, width, 1))
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                       padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                       padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                       padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                       padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)
model = Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(data, encoded_Y, batch_size=batch_size, epochs=num_epochs, validation_split=0.1,
                 callbacks=[MinAccJournal()])

data_test, labels_test = var6.gen_data(size=100)
data_test /= np.max(data_test)
encoder = LabelEncoder()
encoder.fit(labels_test)
encoded_Y = encoder.transform(labels_test)
encoded_Y = np.reshape(encoded_Y, (100, 1))
encoded_Y = np_utils.to_categorical(encoded_Y, num_classes)

np.savetxt("table.txt", np.asarray(table), fmt='%s', delimiter=' ')

model_json = model.to_json()
with open("model_5.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_5.h5")
