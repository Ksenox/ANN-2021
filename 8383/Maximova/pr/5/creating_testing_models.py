import pandas
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

cols_index = 6
#извлечение данных
dataframe_train = pandas.read_csv("data_train.csv", header=None)
dataset_train = dataframe_train.values
train_data = dataset_train[:, 0:cols_index]
train_targets = dataset_train[:, cols_index]

dataframe_test = pandas.read_csv("data_test.csv", header=None)
dataset_test = dataframe_test.values
test_data = dataset_test[:, 0:cols_index]
test_targets = dataset_test[:, cols_index]

#нормализация данных
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = train_data - mean
train_data = train_data / std
test_data = test_data - mean
test_data = test_data / std

np.savetxt(fname="normalized_data_test.csv", X=test_data,  delimiter=',', header='Нормализированные тестовые данные')

#входные данные
input_layer = Input(shape=(cols_index,), name='input_layer')

#кодирование
encoded = Dense(pow(2, 6), activation='relu')(input_layer)
encoded = Dense(pow(2, 5), activation='relu')(encoded)
encoded = Dense(pow(2, 4), activation='relu')(encoded)
#закодированные данные
encoded_output = Dense(pow(2, 3), name='encoded_output')(encoded)

#декодирование
decoded = Dense(pow(2, 4), activation='relu', name='decoder1')(encoded_output)
decoded = Dense(pow(2, 5), activation='relu', name='decoder2')(decoded)
decoded = Dense(pow(2, 6), activation='relu', name='decoder3')(decoded)
#декодированные данные
decoded_output = Dense(cols_index, name='decoded_output')(decoded)

#регрессия
regression = Dense(pow(2, 4), activation='relu')(encoded_output)
regression = Dense(pow(2, 4), activation='relu')(regression)
regression = Dense(pow(2, 3), activation='relu')(regression)
regression_output = Dense(1, name='regression_output')(regression)

#создание модели с 2мя выходами
#regression_output - должны получить значение 7 признака по 1-6
#decoded_output - должны получить исходные данные
combo_model = Model(inputs=input_layer, outputs=[regression_output, decoded_output])
combo_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
combo_model.fit(train_data, [train_targets, train_data], epochs=100, batch_size=20, verbose=0)

#РАЗДЕЛЕНИЕ НА 3 МОДЕЛИ
#входные данные -> результат регрессии
regression_model = Model(inputs=input_layer, outputs=regression_output)
regression_test = regression_model.predict(test_data)
#что должно быть и что предсказывает модель
test_targets.shape = len(test_targets), 1
data = np.hstack((test_targets, regression_test))
np.savetxt(fname="regression_test.csv", X=data, delimiter=',', header='Должно быть/Предсказала модель')

regression_model.save('regression_model.h5')
del regression_model

#вх. данные -> закодированные данные
encoded_model = Model(inputs=input_layer, outputs=encoded_output)
encoded_test = encoded_model.predict(test_data)

np.savetxt(fname="encoded_test.csv", X=encoded_test,  delimiter=',', header='Закодированные данные - ТЕСТ')

encoded_model.save('encoded_model.h5')
del encoded_model

#зак. данные -> декод. данные (нужно отп-ть закодированные данные!)
input_layer = Input(shape=(pow(2, 3),), name='input_decoded')
decoder = combo_model.get_layer('decoder1')(input_layer)
decoder = combo_model.get_layer('decoder2')(decoder)
decoder = combo_model.get_layer('decoder3')(decoder)
decoder_out = combo_model.get_layer('decoded_output')(decoder)

decoded_model = Model(inputs=input_layer, outputs=decoder_out)
decoded_test = decoded_model.predict(encoded_test)

np.savetxt(fname="decoded_test.csv", X=decoded_test, delimiter=',', header='Декодированные данные - ТЕСТ')

decoded_model.save('decoded_model.h5')
del decoded_model