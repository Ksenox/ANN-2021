import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


n = 3000
X = np.random.normal(3, 10, n)
e = np.random.normal(0, 0.3, n)
data = np.array([X ** 2, np.sin(X // 2), X - 3, -X, np.abs(X), (X ** 3) // 4])
data += e
data = np.transpose(data)
target = np.array(np.cos(2*X) + e)
train_size = 500

train_data = data[:train_size, :]
test_data = data[train_size:, :]
train_labels = target[:train_size]
test_labels = target[train_size:]

np.savetxt("trainData.csv", train_data, delimiter=';')
np.savetxt("testData.csv", test_data, delimiter=';')
np.savetxt("trainLabels.csv", train_labels, delimiter=';')
np.savetxt("testLabels.csv", test_labels, delimiter=';')


train_data -= train_data.mean(axis=0)
train_data /= train_data.std(axis=0)
test_data -= train_data.mean(axis=0)
test_data /= train_data.std(axis=0)

# Создание модели

main_input = Input(shape=(6,))
encoded = Dense(48, activation='relu')(main_input)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(2)(encoded)

decoded = Dense(32, activation='relu', name='d1')(encoded)
decoded = Dense(48, activation='relu', name='d2')(decoded)
decoded = Dense(6, name='d_out')(decoded)

regression = Dense(50, activation='relu')(encoded)
regression = Dense(50, activation='relu')(regression)
regression = Dense(50, activation='relu')(regression)
regression = Dense(1, name='p_out')(regression)

model = Model(main_input, outputs=[decoded, regression])
model.compile(optimizer='adam', loss='mse', loss_weights=[0.8, 0.5])
model.fit(train_data, [train_data, train_labels], epochs=50, batch_size=50, validation_split=0.1)
model.evaluate(test_data, [test_data, test_labels])

encodingModel = Model(main_input, encoded)
regressionModel = Model(main_input, regression)

encoded_input = Input(shape=(2,))
decoding_layer = model.get_layer("d1")(encoded_input)
decoding_layer = model.get_layer("d2")(decoding_layer)
decoding_layer = model.get_layer("d_out")(decoding_layer)
decodingModel = Model(encoded_input, decoding_layer)

encodingModel.save('encodingModel.h5')
regressionModel.save('regressionModel.h5')
decodingModel.save('decodingModel.h5')

encodedResult = encodingModel.predict(test_data)
decodedResult = decodingModel.predict(encodedResult)
regressionResult = regressionModel.predict(test_data)

np.savetxt("encodedResult.csv", encodedResult, delimiter=';')
np.savetxt("decodedResult.csv", decodedResult, delimiter=';')
np.savetxt("regressionResult.csv", regressionResult, delimiter=';')
