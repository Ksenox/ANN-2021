from keras.models import load_model
import numpy as np


test_data = np.genfromtxt('test.csv', delimiter=';')
test_x = np.reshape(test_data[:, 0], (len(test_data), 1))

codded = test_x * 2

model = load_model('encoding_model.h5')
encoding_prediction = model.predict(test_x)

np.savetxt('encoding_results.csv', np.hstack((codded, encoding_prediction)), delimiter=';')
