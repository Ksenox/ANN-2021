import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# Используемые для генерации данных формулы
def f1(X, e):
    return np.cos(X) + e


def f2(X, e):
    return -X + e


def f3(X, e):
    return np.sin(X) * X + e


def f4(X, e):
    return np.sqrt(np.abs(X)) + e


def f5(X, e):
    return X ** 2 + e


def f6(X, e):
    return -np.abs(X) + 4


def f7(X, e):
    return X - (X ** 2) / 5 + e


# Генерация датасета
def generate_data(train_size=500, test_size=100):
    rng = np.random.default_rng()
    m = 0
    sd_x = 10
    sd_e = 0.3
    size = train_size + test_size
    X = rng.normal(m, sd_x, size)
    e = rng.normal(m, sd_e, size)
    data = np.asarray([f2(X, e), f3(X, e), f4(X, e), f5(X, e), f6(X, e), f7(X, e)]).transpose()
    labels = np.asarray(f1(X, e))
    train_data = data[:train_size, :]
    test_data = data[train_size:, :]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]
    np.savetxt("train_data.csv", train_data, delimiter=';')
    np.savetxt("test_data.csv", test_data, delimiter=';')
    np.savetxt("train_labels.csv", train_labels, delimiter=';')
    np.savetxt("test_labels.csv", test_labels, delimiter=';')


# Генерация данных
generate_data()

# Загрузка данных
train_data = np.genfromtxt('train_data.csv', delimiter=';')
test_data = np.genfromtxt('test_data.csv', delimiter=';')
train_labels = np.genfromtxt('train_labels.csv', delimiter=';')
test_labels = np.genfromtxt('test_labels.csv', delimiter=';')

# Нормализация данных
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

# Создание модели

inputs = keras.Input(shape=(6,))

# Кодирование данных
encoded = layers.Dense(32, activation='relu')(inputs)
encoded = layers.Dense(18, activation='relu')(encoded)
encoded = layers.Dense(2)(encoded)

# Декодирование данных
decoded = layers.Dense(18, activation='relu', name="dl_1")(encoded)
decoded = layers.Dense(32, activation='relu', name="dl_2")(decoded)
decoded = layers.Dense(6, name="decoding")(decoded)

# Регрессия
regression = layers.Dense(40, activation='relu')(encoded)
regression = layers.Dense(56, activation='relu')(regression)
regression = layers.Dense(32, activation='relu')(regression)
regression = layers.Dense(32, activation='relu')(regression)
regression = layers.Dense(1, name="regression")(regression)

# Определение и компиляция модели
model = keras.Model(inputs, outputs=[regression, decoded])
model.compile(optimizer='adam', loss='mse', metrics=["mae"], loss_weights=[1, 0.8])

# Обучение модели
model.fit(train_data, [train_labels, train_data], epochs=100, batch_size=50, validation_split=0.1, verbose=0)

# Оценка модели
model.evaluate(test_data, [test_labels, test_data])

# Разбиение модели на три
encoding_model = keras.Model(inputs, encoded)
regression_model = keras.Model(inputs, regression)

encoded_input = keras.Input(shape=(2,))
decoding_layer = model.get_layer("dl_1")(encoded_input)
decoding_layer = model.get_layer("dl_2")(decoding_layer)
decoding_layer = model.get_layer("decoding")(decoding_layer)
decoding_model = keras.Model(encoded_input, decoding_layer)

# Сохранение моделей
encoding_model.save('encoding_model.h5')
regression_model.save('regression_model.h5')
decoding_model.save('decoding_model.h5')

# Кодирование данных
encoded_data = encoding_model.predict(test_data)
np.savetxt("encoded_data.csv", encoded_data, delimiter=';')

# Декодирование
decoded_data = decoding_model.predict(encoded_data)
np.savetxt("decoded_data_normalized.csv", decoded_data, delimiter=';')
np.savetxt("test_data_normalized.csv", test_data, delimiter=';')
decoded_data *= std
decoded_data += mean
np.savetxt("decoded_data_denormalized.csv", decoded_data, delimiter=';')

# Регрессия
regression_predictions = regression_model.predict(test_data)
regression_predictions = regression_predictions.reshape((regression_predictions.shape[0]))
combined_data = np.asarray([test_labels, regression_predictions]).transpose()
np.savetxt("regression_data.csv", combined_data, delimiter=';')
