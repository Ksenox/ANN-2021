import numpy as np
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Average, Dropout, Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from string import punctuation

top_words = 10000

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

'''len_data = int(len(data)*0.8)
X_train = data[:len_data]
y_train = targets[:len_data]
X_test = data[len_data:]
y_test = targets[len_data:]'''
X_train = data[:100]
y_train = targets[:100]
X_test = data[100:110]
y_test = targets[100:110]

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

def simple_model():
	embedding_vector_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def conv_model():
	embedding_vector_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.5))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def ensemble_predict(models, X):
    return np.round(sum([one_model.predict(X) for one_model in models])/len(models)).flatten()

def ensemble_evaluate(models, X, Y):
	count_good = np.count_nonzero(ensemble_predict(models, X) == Y)
	return count_good / len(Y)

models = [simple_model(), conv_model()]
for i in range(len(models)):
	models[i].fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

score = ensemble_evaluate(models, X_test, y_test)
print("Accuracy: %.2f%%" % (score*100))

text = open('text.txt').read()
print('введён текст: ', text)

def user_text(text):
	for i in punctuation:
		text = text.replace(i,' ')
	text = text.lower().split()
	index = imdb.get_word_index()
	coded_text = [index.get(i)+3 for i in text]
	return sequence.pad_sequences([coded_text], maxlen=max_review_length)

prediction = ensemble_predict(models, user_text(text))
print(prediction)

if prediction >= 0.5: print('pisitive')
else: print('negative')

