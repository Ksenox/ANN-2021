import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import tensorflow as tf
from keras.datasets import imdb
from keras.models import model_from_json
import re
#mpl.use('TKAgg')

top_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
targets = np.array(targets).astype("float32")


num_models = 5
max_review_length = 500
num_val_samples = len(data) // num_models
embedding_vecor_length = 32

models = []

# for i in range(num_models):
#     val_data = data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = targets[i * num_val_samples: (i + 1) * num_val_samples]
#     tmp = data[(i + 1) * num_val_samples:]
#     partial_train_data = np.concatenate([data[:i * num_val_samples],
#                                          data[(i + 1) * num_val_samples:]], axis=0)
#     partial_train_targets = np.concatenate([targets[:i * num_val_samples],
#                                             targets[(i + 1) * num_val_samples:]], axis=0)
#
#     partial_train_data = sequence.pad_sequences(partial_train_data, maxlen=max_review_length)
#     val_data = sequence.pad_sequences(val_data, maxlen=max_review_length)
#
#     model = Sequential()
#     model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#     model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.5))
#     model.add(LSTM(20, return_sequences=True, dropout=0.5))
#     model.add(Dropout(0.5))
#     model.add(LSTM(30))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
#               epochs=3, batch_size=64, )
#     models.append(model)
#     model_json = model.to_json()
#     with open(str(i) + '.json', "w") as json_file:
#         json_file.write(model_json)
#     model.save_weights(str(i) + '.h5')

def load_model(str):
    json_file = open(str + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(str + '.h5')
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


for i in range(num_models):
    model = load_model(str(i))
    models.append(model)


def text_prepare(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = text.split(' ')
    index = imdb.get_word_index()
    coded = []

    for word in text:
        ind = index.get(word)
        if ind is not None and ind < top_words:
            coded.append(ind + 3)
    coded = np.array([coded])
    return sequence.pad_sequences(coded, maxlen=max_review_length)


def user_input():
    print("Type file name:\n")
    filename = input()
    f = open(filename+".txt", "r")
    text = f.read()
    prediction(text_prepare(text))
    return 0


def prediction(coded):
    res = np.zeros(num_models)
    for i in range(num_models):
        res[i] = models[i].predict(coded)

    mean_res = np.mean(res)
    print(mean_res)
    if mean_res >= 0.5:
        print('good')
    else:
        print('bad')


user_input()
