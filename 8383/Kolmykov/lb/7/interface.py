import re
from keras.datasets import imdb
from keras.models import load_model
import numpy as np
from keras.preprocessing import sequence


def interface(models, vec_len, max_len):
    index = imdb.get_word_index()
    while (True):
        print('Input text (or "stop"):')
        text = input()
        text = clear_text(text)
        if text == 'stop':
            break

        ind_arr = np.array([get_indexes_from_text(text, index, vec_len)])
        ind_arr = sequence.pad_sequences(ind_arr, maxlen=max_len)
        res = predict(models, ind_arr)
        if res >= 0.5:
            print('Positive')
        else:
            print('Negative')


def predict(models, data):
    res_arr = np.zeros(len(models))
    for i in range(len(models)):
        res_arr[i] = models[i].predict(data)
    print(res_arr)
    return np.mean(res_arr)


def clear_text(text):
    text = re.sub(r'([^A-z ]|[\[\]])', '', text).lower()
    text = text.strip()
    text = re.sub(r'  +', ' ', text)
    return text


def get_indexes_from_text(text, index, vec_len):
    text_arr = text.split(" ")
    ind_arr = []
    for word in text_arr:
        num = index.get(word)
        if num is not None and num < vec_len:
            ind_arr.append(num + 3)
    return ind_arr


models = []
for i in range(5):
    model = load_model('model' + str(i) + '.h5')
    models.append(model)
top_words = 10000
max_len = 500
interface(models, top_words, max_len)
