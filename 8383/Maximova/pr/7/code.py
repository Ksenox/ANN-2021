from var5 import gen_sequence
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt

# создание датасета
def gen_data_from_sequence(seq_len=1000, lookback=10):
    seq = gen_sequence(seq_len)

    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])

    return (past, future)

# разбить датасет на обучающую, контрольную и тестовую выборки
def getDatasets(all_data, all_labels):
    train_size = (len(all_data) // 10) * 7  # 70%
    val_size = (len(all_data) - train_size) // 2  # 15%

    train_data, train_labels = all_data[:train_size], all_labels[:train_size]
    val_data, val_labels = all_data[train_size:train_size+val_size], all_labels[train_size:train_size+val_size]
    test_data, test_labels = all_data[train_size+val_size:], all_labels[train_size+val_size:]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def buildModel():
    model = models.Sequential()

    model.add(layers.GRU(48, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
    model.add(layers.LSTM(48, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.35, recurrent_dropout=0.2))
    model.add(layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
    model.add(layers.Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )
    return model

def plotLoss(loss, val_loss, epochs):
    plt.plot(epochs, loss, label="Training loss", linestyle='--', linewidth=2, color='green')
    plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red')

    plt.title('Training and Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()
    plt.savefig("Loss.png", format="png", dpi=240)
    plt.show()

def plotPredict(predict_res, test_Y, epochs):
    plt.clf()

    plt.plot(epochs, predict_res, label="Predicted sequences", linestyle='--', linewidth=2, color='green')
    plt.plot(epochs, test_Y, 'b', label="Generated sequences", color='red')

    plt.title('Predicted and Generated sequences')
    plt.xlabel("x")
    plt.ylabel("Sequence")

    plt.legend()
    plt.grid()
    plt.savefig("Predict.png", format="png", dpi=240)
    plt.show()

data, labels = gen_data_from_sequence()
train_X, train_Y, val_X, val_Y, test_X, test_Y = getDatasets(data, labels)

model = buildModel()
history = model.fit(
    train_X, train_Y,
    epochs=70,
    validation_data=(val_X, val_Y)
)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plotLoss(loss, val_loss, epochs)

predict_res = model.predict(test_X)
epochs = range(len(predict_res))
plotPredict(predict_res, test_Y, epochs)