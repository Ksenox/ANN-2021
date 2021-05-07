import numpy as np 
import random
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers

def func(i):
    return (i % 16 + 1) / 16

def gen_sequence(seq_len = 1000):
    seq = [math.cos(i/10) * func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)),seq)
    plt.show()

def gen_data_from_sequence(seq_len = 1000, lookback = 10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i,i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback,len(seq))])
    return (past, future)

train_size = int(1000*0.8)
X, Y = gen_data_from_sequence()
train_X, val_X, test_X = X[:train_size], X[train_size:train_size+(1000-train_size)//2], X[train_size+(1000-train_size)//2:]
train_Y, val_Y, test_Y = Y[:train_size], Y[train_size:train_size+(1000-train_size)//2], Y[train_size+(1000-train_size)//2:]

model = Sequential()
model.add(layers.GRU(64, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(layers.LSTM(64, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
model.add(layers.GRU(64, input_shape=(None, 1), recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')
history = model.fit(train_X, train_Y, epochs=50, validation_data=(val_X, val_Y))

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss)
plt.plot(range(len(val_loss)), val_loss)
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predicted_res = model.predict(test_X)
pred_length = range(len(predicted_res))
plt.plot(pred_length, predicted_res)
plt.plot(pred_length, test_Y)
plt.legend(['predict', 'real'], loc='upper left')
plt.show()