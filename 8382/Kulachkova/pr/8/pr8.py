import var5
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn.preprocessing import LabelEncoder


# # Callback
class LowestAccuracyLogger(keras.callbacks.Callback):
    def __init__(self, interval, train_data, train_labels, accuracy_function):
        super(LowestAccuracyLogger, self).__init__()
        self.interval = interval
        self.train_data = train_data
        self.train_labels = train_labels
        self.acc_fn = accuracy_function
        self.table = {"epoch": [], "ind": [], "class": [], "acc": [], "loss": []}
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0 or epoch == self.params["epochs"] - 1:
            predictions = self.model.predict(self.train_data)
            acc = self.acc_fn(self.train_labels, predictions)
            loss_fn = keras.losses.get(self.model.loss)
            losses = np.asarray(loss_fn(self.train_labels, predictions))
            min_acc_ind = np.argmin(acc)
            min_acc = acc.flatten()[min_acc_ind]
            min_acc_loss = losses[min_acc_ind]
            min_acc_class = self.train_labels.flatten()[min_acc_ind]
            self.table["epoch"].append(epoch)
            self.table["ind"].append(min_acc_ind)
            self.table["class"].append(int(min_acc_class))
            self.table["acc"].append(min_acc)
            self.table["loss"].append(min_acc_loss)

    def on_train_end(self, logs=None):
        df = pd.DataFrame(data=self.table)
        df.to_csv("LowestAccuracyLog.csv")


def accuracy(y_true, y_pred):
    return 1 - abs(y_pred - y_true)


# Подготовка данных
data, labels = var5.gen_data()
sz = data.shape[0]
img_size = data.shape[1]
labels = labels.reshape(sz)
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels).reshape(sz, 1)
data = data.reshape(sz, img_size**2)
data = np.hstack((data, labels))
rng = np.random.default_rng()
rng.shuffle(data)
labels = data[:, -1].reshape(sz, 1)
data = data[:, :data.shape[1]-1].reshape(sz, img_size, img_size, 1)
data /= np.max(data)
tr_sz = int(sz*0.9)
train_data = data[:tr_sz, :]
train_labels = labels[:tr_sz, :]
test_data = data[tr_sz:, :]
test_labels = labels[tr_sz:, :]

# Построение модели
batch_size = 10
num_epochs = 12
kernel_size = 3
pool_size = 2
conv_depth_1 = 16
conv_depth_2 = 32
hidden_size = 100

inp = Input(shape=(img_size, img_size, 1))
conv_1 = Convolution2D(conv_depth_1, kernel_size, padding='same', activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
conv_2 = Convolution2D(conv_depth_2, kernel_size, padding='same', activation='relu')(pool_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
flat = Flatten()(pool_2)
hidden = Dense(hidden_size, activation='relu')(flat)
out = Dense(1, activation='sigmoid')(hidden)

model = Model(inp, out)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1, callbacks=[LowestAccuracyLogger(4, train_data, train_labels, accuracy)])

