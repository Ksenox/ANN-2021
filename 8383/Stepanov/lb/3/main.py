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

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 5
num_val_samples = len(train_data) // k
num_epochs = 75
all_scores = []
all_loss = []

H = []
val_loss = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]  # берем первую из 4х частей даты
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0) # получаем кусок даты без первой части
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model() # создаем модель
    H.append(model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(partial_train_data, partial_train_targets)))
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    all_loss.append(val_mse)

print(all_scores)
print(np.mean(all_scores))

#Получение ошибки и точности в процессе обучения

loss = []
mae = []

val_loss = []
val_mae = []


epochs = range(1, num_epochs+1)

for i in range(num_epochs):

    counter_loss = 0.0
    counter_mae = 0.0
    counter_val_loss = 0.0
    counter_val_mae = 0.0

    for h in H:
        counter_loss += h.history['loss'][i]
        counter_mae += h.history['mae'][i]
        counter_val_loss += h.history['val_loss'][i]
        counter_val_mae += h.history['val_mae'][i]


    loss.append(counter_loss / k)
    mae.append(counter_mae / k)
    val_loss.append(counter_val_loss / k)
    val_mae.append(counter_val_mae / k)


#Построение графика ошибки
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Training val_loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#Построение графика точности
plt.clf()
plt.plot(epochs, mae, 'b', label='Training mae')
plt.plot(epochs, val_mae, 'bo', label='Training val_mae')
plt.title('Training mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
#Получение и вывод результатов на тестовом наборе
results = model.evaluate(test_data, test_targets)
print("Results on test data =", results)