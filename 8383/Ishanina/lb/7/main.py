import numpy as np
from keras.datasets import imdb
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import string

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=10000)
data = np.concatenate((X_train, Y_test), axis=0)
targets = np.concatenate((Y_train, Y_test), axis=0)
max_review_length = 500
top_words = 10000
embedding_vector_length = 32
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


def build_models():
    models = []
    model1 = Sequential()
    model1.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model1.add(LSTM(100))
    model1.add(Dropout(0.4))
    model1.add(Dense(64, activation='relu'))
    model1.add(Dropout(0.2))
    model1.add(Dense(1, activation='sigmoid'))
    models.append(model1)

    model2 = Sequential()
    model2.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model2.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(Dropout(0.3))
    model2.add(LSTM(150))
    model2.add(Dropout(0.3))
    model2.add(Dense(1, activation='sigmoid'))
    models.append(model2)
    return models


def models_fit(models):
    for i, model in enumerate(models):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        H = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2, batch_size=64)
        scores = model.evaluate(X_test, Y_test, verbose=0)
        model.save('model' + str(i) + '.h5')
        print("accuracy value: %.2f%%" % (scores[1] * 100))
        epochs = range(1, len(H.history['loss']) + 1)
        plt.plot(epochs, H.history['loss'], label='Training loss')
        plt.plot(epochs, H.history['val_loss'], label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.clf()
        plt.plot(epochs, H.history['accuracy'], label='Training accuracy')
        plt.plot(epochs, H.history['val_accuracy'], label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()



def models_ensembling():
    model1 = load_model("model0.h5")
    model2 = load_model("model1.h5")
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)
    predictions = np.divide(np.add(predictions1, predictions2), 2)
    targets = np.reshape(Y_test, (Y_test.shape[0], 1))
    predictions = np.greater_equal(predictions, np.array([0.5]))
    predictions = np.logical_not(np.logical_xor(predictions, targets))
    acc = predictions.mean()
    print("accuracy of ensembling  %s" % acc)


def loading_text(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    arr = [w.translate(table) for w in words]
    arr_low = []
    for w in arr:
        arr_low.append(w.lower())
    print(arr_low)
    indexes = imdb.get_word_index()
    encoded = []
    for w in arr_low:
        if w in indexes and indexes[w] < 10000:
            encoded.append(indexes[w]+3)
    data = np.array(encoded)
    test = sequence.pad_sequences([data], maxlen=max_review_length)
    model1 = load_model("model0.h5")
    model2 = load_model("model1.h5")
    results = []
    results.append(model1.predict(test))
    results.append(model2.predict(test))
    print(results)
    result = np.array(results).mean(axis=0)
    print(result)


models = build_models()
models_fit(models)
models_ensembling()
loading_text("review.txt")