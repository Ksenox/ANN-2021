from var3 import gen_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras import callbacks
import numpy as np
import matplotlib.pyplot as plt

def gen_data_from_sequence(seq_len=1000, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return (past, future)


data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

model = Sequential()
model.add(layers.GRU(32, recurrent_activation='sigmoid', input_shape=(None,1), return_sequences=True))
model.add(layers.LSTM(48, activation='relu',input_shape=(None,1), return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.GRU(32, recurrent_activation='sigmoid', input_shape=(None,1), recurrent_dropout=0.25, return_sequences=True))
model.add(layers.LSTM(32, activation='relu',input_shape=(None,1), dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')

print(model.summary())
history = model.fit(train_data, train_res, epochs=75, validation_data=(val_data, val_res), verbose=0)

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length,predicted_res)
plt.plot(pred_length,test_res)
plt.show()
