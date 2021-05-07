from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def func(i):
    return ((i % 20) + 1) / 20

def gen_sequence(seq_len = 1000):
    seq = [abs(math.sin(i/20)) + func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)),seq)
    plt.show()

draw_sequence()

def gen_data_from_sequence(seq_len = 1006, lookback = 10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i,i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback,len(seq))])
    return (past, future)

data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

model = Sequential()
model.add(layers.GRU(128,recurrent_activation='sigmoid',input_shape=(None,1),return_sequences=True))
model.add(layers.LSTM(128,activation='relu',input_shape=(None,1),return_sequences=True,dropout=0.4))
model.add(layers.GRU(32,input_shape=(None,1),recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')
history = model.fit(train_data,train_res,epochs=50,validation_data=(val_data, val_res))

res = model.predict(test_data)

plt.plot(range(len(res)),test_res)
plt.plot(range(len(res)),res)
plt.legend(['test', 'predicted'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()