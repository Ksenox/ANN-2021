import numpy as np
import tensorflow.keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential     
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

from var5 import gen_data


class AccuracyDrawCallBack(tensorflow.keras.callbacks.Callback):

    def __init__(self, x_test, y_test):
        super(AccuracyDrawCallBack, self).__init__()
        self.Data = x_test
        self.Labels = y_test
        self.count = []

    def on_epoch_end(self, epoch, logs=None):
        count_in_epoch = 0
        predicted = self.model.predict(self.Data)
        for i in range(len(predicted)):
            if abs(self.Labels[i] - predicted[i]) >= 0.1:
                count_in_epoch += 1
        self.count.append(count_in_epoch)

    def on_train_end(self, logs=None):
        print(self.count)
        plt.plot(self.count, color='pink')
        plt.ylabel('Number of observations with accuracy < 90%')
        plt.xlabel('Epochs')
        plt.savefig('graphic.png')
        plt.show()




x, y = gen_data()
x = np.asarray(x)
y = np.asarray(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train = x_train.reshape(x_train.shape[0], 50, 50, 1)
x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
encoder.fit(y_test)
y_test = encoder.transform(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

H = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test), callbacks=[AccuracyDrawCallBack(x_test, y_test)])

model.evaluate(x_test, y_test)

