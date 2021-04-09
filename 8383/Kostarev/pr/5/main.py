import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

train = np.genfromtxt("train.csv", delimiter=";")
validation = np.genfromtxt("validation.csv", delimiter=";")

train_data = np.reshape(train[:,:6], (len(train), 6))
train_labels = np.reshape(train[:,6], (len(train), 1))

validation_data = np.reshape(validation[:,:6], (len(validation), 6))
validation_labels = np.reshape(validation[:,6], (len(validation), 1))

encoding_dim = 4
layer_input_main = Input(shape=(6,), name="input_main")

layer_encoded = Dense(encoding_dim * 6, activation="relu")(layer_input_main)
layer_encoded = Dense(encoding_dim * 3, activation="relu")(layer_encoded)
layer_encoded_output = Dense(encoding_dim, name="output_encoded")(layer_encoded)

layer_input_decoded = Input(shape=(encoding_dim,), name="input_decoding")
layer_decoded = Dense(encoding_dim * 3, activation="relu")(layer_input_decoded)
layer_decoded = Dense(encoding_dim * 6, activation="relu")(layer_decoded)
layer_decoded_output = Dense(6, name="output_decoded")(layer_decoded)

layer_input_regression = Input(shape=(encoding_dim,), name="input_regression")
layer_regression = Dense(encoding_dim * 6, activation="relu")(layer_input_regression)
layer_regression = Dense(encoding_dim * 4, activation="relu")(layer_regression)
layer_regression_output = Dense(1, name="output_regression")(layer_regression)

encoder = Model(layer_input_main, layer_encoded_output, name="encoder")
decoder = Model(inputs=[layer_input_decoded], outputs=[layer_decoded_output], name="decoder")
regression = Model(inputs=[layer_input_regression], outputs=[layer_regression_output], name="regression")

model_main = Model(inputs=[layer_input_main], outputs=[decoder(encoder(layer_input_main)), regression(encoder(layer_input_main))], name="model_main")

model_main.compile(optimizer='rmsprop', loss='mse', metrics='mae')
model_main.fit([train_data], [train_data, train_labels], epochs=300, batch_size=15, validation_split=0)

test_index = np.random.randint(0, len(validation_data) - 1)
test_data = np.reshape(validation_data[test_index,:], (1, 6))
test_label = validation_labels[test_index,:]

print("Test data:", test_data)
print("Test label:", test_label)

encoded_data = encoder.predict(test_data)
print("Encoded data:", encoded_data)

regression_data = regression.predict(encoded_data)
print("Regression prediction:", regression_data)

decoded_data = decoder.predict(encoded_data)
print("Decoder prediction:", decoded_data)

encoded_data = encoder.predict(validation_data)
regression_data = regression.predict(encoded_data)
decoded_data = decoder.predict(encoded_data)

np.savetxt("encoded_data.csv", encoded_data, delimiter=";")
np.savetxt("decoded_data.csv", decoded_data, delimiter=";")
np.savetxt("regression_data.csv", np.hstack((regression_data, validation_labels)), delimiter=";")

encoder.save("model_encoder.h5")
decoder.save("model_decoder.h5")
regression.save("model_regression.h5")