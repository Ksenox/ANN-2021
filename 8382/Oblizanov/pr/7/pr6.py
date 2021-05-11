import keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from var7 import gen_sequence


def gen_data_from_sequence(seq_len=1000, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return (past, future)


data, labels = gen_data_from_sequence()
size = len(data)
tsz = size // 10 * 6
vsz = size // 10 * 2
train_data, train_labels = data[:tsz], labels[:tsz]
val_data, val_labels = data[tsz:tsz + vsz], labels[tsz:tsz + vsz]
test_data, test_labels = data[tsz + vsz:], labels[tsz + vsz:]

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='sin_pred.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
    )
]

model = Sequential()
model.add(layers.GRU(32, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(layers.LSTM(16, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.25))
model.add(layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.25))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(
    train_data,
    train_labels,
    epochs=20,
    callbacks=callbacks_list,
    validation_data=(val_data, val_labels),
)

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss, 'g', label='Train')
plt.plot(range(len(val_loss)), val_loss, 'c', label='Validation')
plt.grid()
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()

predicted = model.predict(test_data)
x = range(len(predicted))
plt.plot(x, predicted, 'g', label='Predicted')
plt.plot(x, test_labels, 'c', label='Generated')
plt.title('Sequence')
plt.xlabel('x')
plt.ylabel('Sequence')
plt.grid()
plt.legend()
plt.savefig('sin.png', format='png', dpi=250)
