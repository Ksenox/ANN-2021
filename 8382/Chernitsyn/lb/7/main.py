import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Conv1D,MaxPooling1D,Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

top_words = 10000

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

max_review_length = 500
training_data = sequence.pad_sequences(training_data, maxlen=max_review_length)
testing_data = sequence.pad_sequences(testing_data, maxlen=max_review_length)
embedding_vecor_length = 32

def build_model_1():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model_2():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def vectorize(sequences, dimension=500):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def test_text(filename):
    f = open(filename, 'r')
    open_text = f.read()
    # print(open_text)
    index = imdb.get_word_index()
    text = []
    for i in open_text:
        if i in index and index[i] < 500:
            text.append(index[i])

    text = vectorize([text])
    return text


def final_pred(model_1,model_2,text):
    pred_1 = model_1.predict(text)
    pred_2 = model_2.predict(text)
    result = 0.5*(pred_1+pred_2)
    if result >= 0.5:
        print('Отзыв хороший')
    else:
        print('Отзыв плохой')

def evaluation(model_1,model_2,testing_data):
    scores_1 = model_1.predict(testing_data)
    scores_2 = model_2.predict(testing_data)
    res = 0.5*(scores_1+scores_2)
    res = round(res)
    return res

if __name__ == '__main__':
    model_1 = build_model_1()
    model_1.fit(training_data, training_targets, validation_data=(testing_data, testing_targets), epochs=3, batch_size=64)
    scores_1 = model_1.evaluate(testing_data, testing_targets, verbose=0)
    model_2 = build_model_2()
    model_2.fit(training_data, training_targets, validation_data=(testing_data, testing_targets), epochs=3, batch_size=64)
    scores_2 = model_2.evaluate(testing_data, testing_targets, verbose=0)
    print("Accuracy: %.2f%%" % (scores_1[1] * 100))
    print("Accuracy: %.2f%%" % (scores_2[1] * 100))
    result = evaluation(model_1,model_2,testing_data)

    filename = "text_good.txt"
    print(filename)
    text = test_text(filename)
    final_pred(model_1,model_2,text)