from gen import genFullData
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

INP_SIZE = 6


def encoding():
    main_input = Input(shape=(INP_SIZE,), name='main_input')
    encoded = Dense(64, activation='relu')(main_input)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(INP_SIZE, activation='linear')(encoded)
    return main_input, encoded


def decoding():
    input_encoded = Input(shape=(INP_SIZE,), name='input_encoded')
    decoded = Dense(32, activation='relu',
                    kernel_initializer='normal')(input_encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(INP_SIZE, name="out_dec")(decoded)
    return input_encoded, decoded


def regression(inp):
    predicted = Dense(64, activation='relu',
                      kernel_initializer='normal')(inp)
    predicted = Dense(32, activation='relu')(predicted)
    predicted = Dense(16, activation='relu')(predicted)
    predicted = Dense(1, name="out_main")(predicted)
    return predicted


def createModels():

    main_input, encoded = encoding()
    input_encoded, decoded = decoding()
    predicted = regression(encoded)

    encoded = Model(main_input, encoded, name="encoder_layer")
    decoded = Model(input_encoded, decoded, name="decoder_layer")
    predicted = Model(main_input, predicted, name="regr_layer")

    return encoded, decoded, predicted


def runModels(encoded, decoded, main_model):

    x_train, y_train, x_test, y_test = genFullData(INP_SIZE)

    main_model.compile(optimizer="adam", loss="mse", metrics=['mae'])

    history = main_model.fit(x_train, y_train, epochs=40,
                             batch_size=5, verbose=1, validation_data=(x_test, y_test))

    encoded_result = encoded.predict(x_test)
    decoded_result = decoded.predict(encoded_result)
    main_result = main_model.predict(x_test)

    decoded.save('./model/decoder.h5')
    encoded.save('./model/encoder.h5')
    main_model.save('./model/main.h5')

    return encoded_result, decoded_result, main_result, history


encoded, decoded, main_model = createModels()
encoded_result, decoded_result, main_result, history = runModels(
    encoded, decoded, main_model)

pd.DataFrame(np.round(main_result, 3)).to_csv("./result/main.csv")
pd.DataFrame(np.round(encoded_result, 3)).to_csv("./result/encoded.csv")
pd.DataFrame(np.round(decoded_result, 3)).to_csv("./result/decoded.csv")
