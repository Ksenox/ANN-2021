import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def generate(size):
    x_data = np.zeros((size, 6))
    y_data = np.zeros(size)
    for i in range(size):
        X = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        x_data[i, :] = (
            X ** 2 + X + e,
            np.fabs(X) + e,
            np.sin(X - np.pi / 4) + e,
            np.log(np.fabs(X)) + e,
            - X ** 3 + e,
            -X / 4 + e)
        y_data[i] = -X + e
    return np.array(x_data), np.array(y_data)


x_train, y_train = generate(500)
x_test, y_test = generate(50)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

main_input = Input(shape=(6,), name='model_input')
encoded = Dense(48, activation='relu')(main_input)
encoded = Dense(24, activation='relu')(encoded)
encoded = Dense(4)(encoded)

decoded = Dense(24, activation='relu', name='d1')(encoded)
decoded = Dense(48, activation='relu', name='d2')(decoded)
decoded = Dense(6, name='d_output')(decoded)

regression = Dense(30, activation='relu')(encoded)
regression = Dense(30, activation='relu')(regression)
regression = Dense(30, activation='relu')(regression)
regression = Dense(1, name='p_output')(regression)

full_model = Model(main_input, outputs=[decoded, regression])

# init and learn
full_model.compile(optimizer='adam', loss='mse', loss_weights=[0.5, 0.5])
regr_res = full_model.fit(
    {'model_input': x_train}, {'d_output': x_train, 'p_output': y_train},
    epochs=50,
    batch_size=10,
    validation_data=({'model_input': x_test}, {'d_output': x_test, 'p_output': y_test}))

encoder = Model(main_input, encoded, name="encoder")
decoded_input = Input(shape=(4,))
decoded_p = full_model.get_layer('d1')(decoded_input)
decoded_p = full_model.get_layer('d2')(decoded_p)
decoded_p = full_model.get_layer('d_output')(decoded_p)
decoder = Model(decoded_input, decoded_p)
regression = Model(main_input, regression)
encoded_res = encoder.predict(x_test)
decoded_res = decoder.predict(encoded_res)
regression_res = regression.predict(x_test)

decoder.save('decoder.h5')
encoder.save('encoder.h5')
regression.save('regression.h5')

pd.DataFrame(np.round(x_test, 5)).to_csv("x_test.csv")
pd.DataFrame(np.round(y_test, 5)).to_csv("y_test.csv")
pd.DataFrame(np.round(x_train, 5)).to_csv("x_train.csv")
pd.DataFrame(np.round(y_train, 5)).to_csv("y_train.csv")

pd.DataFrame(np.round(encoded_res, 5)).to_csv("encoded_res.csv")
pd.DataFrame(np.round(decoded_res, 5)).to_csv("decoded_res.csv")
pd.DataFrame(np.round(regression_res, 5)).to_csv("regression_res.csv")