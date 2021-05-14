import numpy as np
from datetime import datetime

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from var1_pr6 import gen_data
import random


class SaveModel(Callback):
    def __init__(self, epochs, prefix, date=datetime.now()):
        self.prefix = str(date.day)+"_"+str(date.month)+"_"+str(date.year)+"_"+prefix+"_"
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs:
            self.model.save(self.prefix + str(epoch))

samples, size = 500, 50
data, labels = gen_data(samples, size)
rand = list(range(len(data)))
random.shuffle(rand)
data = data[rand]
labels = labels[rand]
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)
labels = to_categorical(labels)

data = data.reshape(data.shape[0], size, size, 1)
epochs = [2, 5, 7, 8, 9, 12]
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size, size, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
H = model.fit(data, labels, batch_size=20, epochs=12, validation_split=0.2, callbacks=[SaveModel(epochs, 'model')])

loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['acc']
val_acc = H.history['val_acc']
epochs = range(1, len(loss) + 1)


plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()