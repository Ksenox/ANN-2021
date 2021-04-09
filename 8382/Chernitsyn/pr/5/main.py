import numpy as np
import csv

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


def getData(nrow):
    data = np.zeros((nrow, 6))
    targets = np.zeros(nrow)
    for i in range(nrow):
        x = np.random.normal(-5, 10)
        e = np.random.normal(0, 0.3)
        data[i, :] = (
            (-x) ** 3 + e, np.log(np.fabs(x)) + e, np.sin(3 * x) + e, x + 4 + e, -x + np.sqrt(np.fabs(x)) + e,
            x + e)
        targets[i] = np.exp(x) + e
    return data, targets


def create():
    main_input = Input(shape=(6,), name='mainInput')

    enc = Dense(64, activation='relu')(main_input)
    enc = Dense(32, activation='relu')(enc)
    enc = Dense(6, activation='relu', name="encode")(enc)

    input2 = Input(shape=(6,), name='input_encoded')

    dec = Dense(32, activation='relu')(input2)
    dec = Dense(64, activation='relu')(dec)
    dec = Dense(6, name='decode')(dec)

    pred = Dense(64, activation='relu')(enc)
    pred = Dense(32, activation='relu')(pred)
    pred = Dense(16, activation='relu')(pred)
    pred = Dense(1, name="predict")(pred)

    enc = Model(main_input, enc, name="encoder")
    dec = Model(input2, dec, name="decoder")
    pred = Model(main_input, pred, name="autoencoder")
    return enc, dec, pred, main_input


def write_csv(path, data):
    with open(path, 'w', newline='') as file:
        my_csv = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        try:
            for i in data:
                my_csv.writerow(i)
        except Exception as ex:
            my_csv.writerow(data)


x_train, y_train = getData(150)
x_test, y_test = getData(50)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std

encoder, decoder, autoEncoder, mainInput = create()

autoEncoder.compile(optimizer="adam", loss="mse", metrics=['mae'])
autoEncoder.summary()
autoEncoder.fit(x_train, y_train,
                epochs=50,
                batch_size=1,
                shuffle=True,
                verbose=0,
                validation_data=(x_test, y_test))

encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)
regr = autoEncoder.predict(x_test)

decoder.save('decoder.h5')
encoder.save('encoder.h5')
autoEncoder.save('keras_model.h5')

write_csv('./x_train.csv', x_train)
write_csv('./y_train.csv', y_train)
write_csv('./x_test.csv', x_test)
write_csv('./y_test.csv', y_test)
write_csv('./encoded.csv', encoded_data)
write_csv('./decoded.csv', decoded_data)
write_csv('./regression_predicted.csv', regr)