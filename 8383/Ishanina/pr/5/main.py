import csv
import numpy as np
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model


def writing_to_file(data, filename):
    file = open(filename, 'w', newline='')
    writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in data.tolist():
        if type(i) == float:
            writer.writerow([i])
        else:
            writer.writerow(i)


def generator(size):
    formulas = []
    answer = []
    for i in range(size):
        x = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        formulas.append((x ** 2 + x + e,
                     np.fabs(x) + e,
                     np.sin(x - np.pi / 4) + e,
                     np.log(np.fabs(x)) + e,
                     -1 * x ** 3 + e,
                     -1 * x + e))
        answer.append((-1 * x / 4 + e))
    return np.array(formulas), np.array(answer)


train_size = 400
test_size = 50

train_data, train_labels = generator(train_size)
test_data, test_labels = generator(test_size)

mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std

writing_to_file(train_data, 'train_data.csv')
writing_to_file(train_labels, 'train_labels.csv')
writing_to_file(test_data, 'test_data.csv')
writing_to_file(test_labels, 'test_labels.csv')

# описание энкодера
inputt = Input(shape=(6,), name='input')
encoding1 = Dense(24, activation='relu')(inputt)
encoding2 = Dense(12, activation='relu')(encoding1)
encoding3 = Dense(4)(encoding2)
# описание декодера
decoding1 = Dense(12, activation='relu', name='decoding1')(encoding3)
decoding2 = Dense(24, activation='relu', name='decoding2')(decoding1)
decoding3 = Dense(6, name='decoded_end')(decoding2)
# регрессия
regression1 = Dense(30, activation='relu')(encoding3)
regression2 = Dense(30, activation='relu')(regression1)
regression3 = Dense(30, activation='relu')(regression2)
regression4 = Dense(1, name='predicted_end')(regression3)

special_model = Model(inputt, outputs=[decoding3, regression4])

special_model.compile(optimizer='adam', loss='mse', loss_weights=[0.7, 0.3])
res_regr = special_model.fit(
    {'input': train_data},
    {'decoded_end': train_data, 'predicted_end': train_labels},
    epochs=60,
    batch_size=15,
    validation_data=({'input': test_data}, {'decoded_end': test_data, 'predicted_end': test_labels}))

# модель кодирования данных
encoder = Model(inputt, encoding3)
# регрессионная модель
regression = Model(inputt, regression4)
# модель декодирования данных
decoder_input = Input(shape=(4,))
decoder = special_model.get_layer('decoding1')(decoder_input)
decoder = special_model.get_layer('decoding2')(decoder)
decoder = special_model.get_layer('decoded_end')(decoder)
decoder = Model(decoder_input, decoder)
encoder_predict = encoder.predict(test_data)
decoder_predict = decoder.predict(encoder_predict)
regression_predict = regression.predict(test_data)
writing_to_file(encoder_predict, 'encoder.csv')
writing_to_file(regression_predict, 'regression.csv')
writing_to_file(decoder_predict, 'decoder.csv')
encoder.save('encoder.h5')
decoder.save('decoder.h5')
regression.save('regression.h5')