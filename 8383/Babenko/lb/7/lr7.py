import numpy as np
from keras import Sequential
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.layers import Embedding, Conv1D, Dropout, MaxPooling1D, LSTM, Dense
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=500)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
MAX_REVIEW_LENGTH = 500
EMBEDING_VECOR_LENGTH = 32


def plot_loss(loss, v_loss):
    plt.figure(1, figsize=(8, 5))
    plt.plot(loss, 'b', label='Train')
    plt.plot(v_loss, 'r', label='Validation')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.clf()


def plot_acc(acc, val_acc):
    plt.plot(acc, 'b', label='Train')
    plt.plot(val_acc, 'r', label='Validation')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.clf()


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_REVIEW_LENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_REVIEW_LENGTH)

X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=123)

y_test = np.asarray(y_test).astype("float32")
y_train = np.asarray(y_train).astype("float32")


def encode_review(rev):
    res = []
    for i, el in enumerate(rev):
        el = el.lower()
        delete_el = [',', '!', '.', '?']
        for d_el in delete_el:
            el = el.replace(d_el, '')
        el = el.split()
        for j, word in enumerate(el):
            code = imdb.get_word_index().get(word)
            if code is None or code >= 10000:
                code = 0
            el[j] = code
        res.append(el)
    for i, r in enumerate(res):
        res[i] = sequence.pad_sequences([r], maxlen=MAX_REVIEW_LENGTH)
    res = np.array(res)
    return res.reshape((res.shape[0], res.shape[2]))


def smart_predict(model, model_2, input):
    pred1 = model.predict(input)
    pred2 = model_2.predict(input)
    pred = [(pred1[i] + pred2[i]) / 2 for i in range(len(pred1))]
    return pred


def review(model, model_2, review):
    data = encode_review(review)
    print(smart_predict(model, model_2, data))


model = Sequential()
model.add(Embedding(10000, 32, input_length=500))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model_2 = Sequential()
model_2.add(Embedding(10000, 32, input_length=500))
model_2.add(LSTM(100))
model_2.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

H = model.fit(x_train, y_train, batch_size=128, epochs=2, verbose=1, validation_split=0.1)

acc = model.evaluate(x_test, y_test)
print('Test', acc)
plot_loss(H.history['loss'], H.history['val_loss'])
plot_acc(H.history['accuracy'], H.history['val_accuracy'])

H_2 = model_2.fit(x_train, y_train, batch_size=128, epochs=2, verbose=1, validation_split=0.1)
acc = model_2.evaluate(x_test, y_test)
print('Test', acc)

plot_loss(H_2.history['loss'], H_2.history['val_loss'])
plot_acc(H_2.history['accuracy'], H_2.history['val_accuracy'])

review(model, model_2, [
    "It is bad, i hate it",
    "Really boring and bad, I am dissapointed"
    "It's amazing, fantastic and exiting",
    "Really good, fantastic actors, i like"
])