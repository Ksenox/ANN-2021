# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colaboratory.


import pandas
#model
from keras.layers import Dense
from keras.models import Sequential
#preprocessing
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


dataframe = pandas.read_csv("/content/sonar.all-data.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


model = Sequential()
model.add(Dense(30, input_dim=60, bias_initializer='normal', activation='relu'))
model.add(Dense(15, input_dim=60, bias_initializer='normal', activation='relu'))
model.add(Dense(1, bias_initializer='normal', activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(loss_values)+1)

# "bo" is for "blue dot"
# b is for "solid blue line"
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()  # очистка фигуры
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
