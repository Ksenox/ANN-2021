from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from datetime import datetime
from var7 import gen_data
import matplotlib.pyplot as plt

class special_callback(Callback):
    def __init__(self, val, prefix='number_model', key='val_loss', date=datetime.now()):
        self.val = val
        self.prefix='{}_{}_{}_{}_'.format(date.day, date.month, date.year, prefix)
        self.loss = {}
        self.key = key
        self.index = 0


    def on_train_begin(self, logs=None):
        loss = self.model.evaluate(self.val[0], self.val[1])[0]
        for i in range(1,4):
            self.loss[self.prefix + str(i)] = loss
        for key in self.loss.keys():
            self.model.save(key)

    def on_epoch_end(self, epoch, logs=None):
        for i in range(1, 4):
            if logs.get(self.key) < self.loss[self.prefix + str(i)] and i > self.index:
                self.loss[self.prefix + str(i)] = logs.get(self.key)
                self.model.save(self.prefix + str(i))
                self.index += 1
                break
            elif i <= self.index:
                continue
        if (self.index == 3):
            self.index = 0


    def on_train_end(self, logs=None):
        for (key, loss) in self.loss.items():
            print(key + ' ' + str(loss))

data, labels = gen_data(5000)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train.reshape(x_train.shape[0], 50, 50, 1)
x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)

encoder = LabelEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
y_test = to_categorical(y_test)
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_train = to_categorical(y_train)

model = Sequential()
model.add(Conv2D(16, kernel_size=(7, 7), activation='relu', input_shape=(50, 50, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=12, batch_size=20, validation_split = 0.1, callbacks=[special_callback((x_test, y_test))])
model.evaluate(x_test, y_test)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()


plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()