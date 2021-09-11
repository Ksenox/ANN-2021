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


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

k = 6
num_val_samples = len(train_data) // k
num_epochs = 18
all_scores = []

mean_loss = []
mean_mae = []
mean_val_loss = []
mean_val_mae = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,
                        validation_data=(val_data, val_targets), verbose=0)
    mean_val_mae.append(H.history['val_mae'])
    mean_mae.append(H.history['mae'])
    plt.plot(H.history['mae'], 'g')
    plt.plot(H.history['val_mae'], 'b')
    plt.title('Mean absolute error' + ', i = ' + str(i + 1))
    plt.ylabel('mae')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    mean_val_loss.append(H.history['val_loss'])
    mean_loss.append(H.history['loss'])

    plt.plot(H.history['loss'], 'g')
    plt.plot(H.history['val_loss'], 'b')
    plt.title('Model loss' + ', i = ' + str(i + 1))
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

plt.plot(np.mean(mean_mae, axis=0), 'g')
plt.plot(np.mean(mean_val_mae, axis=0), 'b')
plt.title('Mean model mae')
plt.ylabel('mae')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

plt.plot(np.mean(mean_loss, axis=0), 'g')
plt.plot(np.mean(mean_val_loss, axis=0), 'b')
plt.title('Mean model loss')
plt.ylabel('loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()