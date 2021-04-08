from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import pandas
import numpy as np
train_dataset = pandas.read_csv("train_data.csv", header=None)
train_dataset = train_dataset.values
train_data = train_dataset[:,0:6].astype(float)
train_labels = train_dataset[:,6].astype(float)

test_dataset = pandas.read_csv("test_data.csv", header=None)
test_dataset = test_dataset.values
test_data = test_dataset[:,0:6].astype(float)
test_labels = test_dataset[:,6].astype(float)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

inputs = Input(shape=(6,))

#encoder
output_e1 = Dense(64, activation='relu')(inputs)
output_e2 = Dense(32, activation='relu')(output_e1)
encoder = Dense(4, name="encoder")(output_e2)

#decoder
output_d1 = Dense(32, activation="relu")(encoder)
output_d2 = Dense(64, activation="relu")(output_d1)
decoder = Dense(6, name="decoder")(output_d2)

#regression
output_r1 = Dense(8, activation="relu")(encoder)
output_r2 = Dense(64, activation="relu")(output_r1)
regression = Dense(1, name="regression")(output_r2)



model = Model(inputs=inputs, outputs=[decoder, regression])
model.compile(optimizer="adam", loss='mse', metrics=['mae'])
model.fit(train_data, [train_data, train_labels], epochs=200, batch_size=10)

model_encoder = Model(inputs=inputs, outputs=encoder)
encoder_predict = model_encoder.predict(test_data)
np.savetxt("encoder_test.csv", encoder_predict, delimiter=",")
model.save("model_encoder.h5")


model_decoder = Model(inputs=inputs, outputs=decoder)
decoder_predict = model_decoder.predict(test_data)
np.savetxt("decoder_test.csv", decoder_predict, delimiter=",")
model.save("model_decoder.h5")

model_regression = Model(inputs=inputs, outputs=regression)
regression_predict = model_regression.predict(test_data)
np.savetxt("regression_test.csv", regression_predict, delimiter=",")
model.save("model_regression.h5")

#print(test_data)
#print("----")
#print(decoder_predict)

#print(test_labels)
#print("----")
#print(regression_predict)