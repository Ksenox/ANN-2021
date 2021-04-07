from typing import Tuple

import numpy as np
from keras import Model
from keras.layers import Dense, Input


def read_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_data = np.genfromtxt("data/train.csv", delimiter=',')
    test_data = np.genfromtxt("data/validation.csv", delimiter=',')
    return train_data[:, 0:6], train_data[:, 6], test_data[:, 0:6], test_data[:, 6]

def main():
    # чтение данных
    X_train, y_train, X_test, y_test = read_data()

    # нормализация данных
    mean_ = X_train.mean(axis=0)
    std_ = X_train.std(axis=0)
    X_train -= mean_
    X_test -= mean_
    X_train /= std_
    X_test /= std_

    # создание архитектуры модели
    inputs = Input(shape=(6,))

    encoded = Dense(64, activation="relu")(inputs)
    encoded = Dense(32, activation="relu")(encoded)
    encoded = Dense(16, activation="relu")(encoded)
    encoded = Dense(8, activation="relu")(encoded)
    encoded = Dense(4, activation="relu")(encoded)
    encoded = Dense(3)(encoded)

    decoded = Dense(4, activation="relu", name="d1")(encoded)
    decoded = Dense(8, activation="relu", name="d2")(decoded)
    decoded = Dense(16, activation="relu", name="d3")(decoded)
    decoded = Dense(32, activation="relu", name="d4")(decoded)
    decoded = Dense(64, activation="relu", name="d5")(decoded)
    decoded = Dense(6, name="decoded")(decoded)

    regression = Dense(64, activation="relu")(encoded)
    regression = Dense(32, activation="relu")(regression)
    regression = Dense(16, activation="relu")(regression)
    regression = Dense(8, activation="relu")(regression)
    regression = Dense(1, name="regression")(regression)

    model = Model(inputs, outputs=[decoded, regression])

    # компиляция и обучение модели
    model.compile(optimizer="adam", loss="mse", metrics="mae")
    model.fit(X_train, [X_train, y_train], epochs=50, batch_size=64, validation_split=0.1, verbose=2)
    model.evaluate(X_test, [X_test, y_test])

    # разбиение модели
    e_model = Model(inputs, encoded)
    inputs_for_decoding = Input(3)
    decoded_copy = model.get_layer("d1")(inputs_for_decoding)
    decoded_copy = model.get_layer("d2")(decoded_copy)
    decoded_copy = model.get_layer("d3")(decoded_copy)
    decoded_copy = model.get_layer("d4")(decoded_copy)
    decoded_copy = model.get_layer("d5")(decoded_copy)
    decoded_copy = model.get_layer("decoded")(decoded_copy)

    d_model = Model(inputs_for_decoding, decoded_copy)
    r_model = Model(inputs, regression)

    # сохранение моделей
    e_model.save("models/encoder.h5")
    d_model.save("models/decoder.h5")
    r_model.save("models/regression.h5")

    # кодирование тестовых данных
    encoded_X_train = e_model.predict(X_test)

    # декодирование тестовых данных
    decoded_X_train_normalized = d_model.predict(encoded_X_train)

    # денормализация декодированных тестовых данных
    decoded_X_train_denormalized = decoded_X_train_normalized * std_
    decoded_X_train_denormalized += mean_

    # предсказывание целевого столбца
    regression_results = r_model.predict(X_test)

    # сохранение закодированных и декодированных данных
    np.savetxt("data/encoded_X_train.csv", encoded_X_train, delimiter=",", fmt="%1.5f")
    np.savetxt("data/decoded_X_train.csv", decoded_X_train_denormalized, delimiter=",", fmt="%1.5f")

    # преобразование формы для сохранения
    y_test = y_test.reshape((y_test.shape[0],1))

    # сохранение результатов регрессии
    np.savetxt("data/regression.csv", np.hstack((y_test, regression_results)), delimiter=",", fmt="%1.5f")


if __name__ == '__main__':
    main()
