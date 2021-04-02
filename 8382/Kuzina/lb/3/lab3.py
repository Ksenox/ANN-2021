import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def draw_plots(history, i):
    m = 'Training and validation loss, k = ' + str(i)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(mae) + 1)
    fig, ax = plt.subplots(2, 2)

    ax[0][0].plot(epochs, loss, 'bo', label='Training loss')
    ax[0][0].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[0][0].set_title(m)
    ax[0][0].set_xlabel('Epochs')
    ax[0][0].set_ylabel('Loss')
    ax[0][0].legend()

    ax[0][1].plot(epochs, mae, 'bo', label='Training mae')
    ax[0][1].plot(epochs, val_mae, 'r', label='Validation mae')
    ax[0][1].set_title('Training and validation mae')
    ax[0][1].set_xlabel('Epochs')
    ax[0][1].set_ylabel('MAE')
    ax[0][1].legend()
    plt.show()


def draw_last(epochs, avg_loss):
    # потери
    plt.plot(epochs, avg_loss, 'r')
    plt.title('Average validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # MAE
    plt.clf()
    plt.plot(epochs, avg_mae, 'r')
    plt.title('Average validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.show()


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


k = 6
num_val_samples = len(train_data) // k
num_epochs = 40
all_val_loss = []
all_val_mae = []

for i in range(k):
    print('Рассматриваем блок №', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Выбрали блок, на оставшихся обучение
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    # Модель
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    # Графики для текущей модели
    #draw_plots(history, i)
    # Сохранение данные о МАЕ и ошибке для среднего
    val_loss = history.history["val_loss"]
    val_mae = history.history["val_mae"]
    all_val_mae.append(val_mae)
    all_val_loss.append(val_loss)

# Графики по средним и оценка модели
all_val_mae = np.asarray(all_val_mae)
avg_mae = all_val_mae.mean(axis=0)
all_val_loss = np.asarray(all_val_loss)
avg_loss = all_val_loss.mean(axis=0)
epochs = range(1, len(avg_mae) + 1)

draw_last(epochs, avg_loss)

print("Срдений MAE: ", np.mean(all_val_mae[:, -1]))
