import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def draw(history):
    loss = history.history['mae']
    val_loss = history.history['val_mae']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training mae')
    plt.plot(epochs, val_loss, 'b', label='Validation mae')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('mae')
    plt.legend()
    plt.show()


def draw_avg(maes, val_maes):
    avg_train = get_avg_list(maes)
    avg_val = get_avg_list(val_maes)
    epochs = range(1, len(avg_train) + 1)
    plt.plot(epochs, avg_train, 'bo', label='Training mae')
    plt.plot(epochs, avg_val, 'b', label='Validation mae')
    plt.title('Average training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('mae')
    plt.legend()
    plt.show()

def get_avg_list(l):
    avg = []
    for i in range(len(l[0])):
        sum = 0
        for j in range(len(l)):
            sum += l[j][i]
        avg.append(sum/len(l))
    return avg


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# print(train_data.shape)
# print(train_targets)
# print(test_data.shape)
# print(test_targets)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 6
num_val_samples = len(train_data) // k
num_epochs = 70
all_scores = []
maes = []
val_maes = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs,
                        batch_size=1, verbose=False, validation_data=(val_data, val_targets))
    draw(history)
    # print(history.history)
    maes.append(history.history['mae'])
    val_maes.append(history.history['val_mae'])
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=False)
    all_scores.append(val_mae)

print(np.mean(all_scores))
print(all_scores)
draw_avg(maes, val_maes)
