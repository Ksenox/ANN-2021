import tensorflow as tf 
device_name = tf.test.gpu_device_name()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPool1D, GRU, Flatten, Embedding
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
from sklearn.metrics import accuracy_score

def plot_history(history, filename=""):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(18,10))
    fig.suptitle('Loss and accuracy')

    ax1.plot(epochs, loss, color="green", label='Training loss')
    ax1.plot(epochs, val_loss, color = "blue", label='Validation loss')
    ax1.legend()

    ax2.plot(epochs, acc, color="green", label='Training acc')
    ax2.plot(epochs, val_acc, color = "blue", label='Validation acc')
    ax2.legend()

    plt.show()

    if (filename != ""):
        plt.savefig(filename)

def print_history(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    print("T loss: {}; V loss: {}; T accuracy: {}; V accuracy: {}"
          .format(loss[-1], val_loss[-1], acc[-1], val_acc[-1]))
    plot_history(history)

def read_text_from_input():
    print("Input string: ")
    words = input()
    words = words.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('\n', ' ').split()
    dictionary = imdb.get_word_index()
    test_x = []
    test_y = []

    for word in words:
        if dictionary.get(word) in range(1, 10000):
            test_y.append(dictionary.get(word) + 3)

    test_x.append(test_y)
    print(test_x)
    result = sequence.pad_sequences(test_x, maxlen=rewiews_count)
    print(predict_all(models, result, True))

def predict_all(models, x_test, load):
    combo = []

    for i, m in enumerate(models):
        if load:
          print(m.predict(x_test, verbose=0))
        combo.append(np.round(m.predict(x_test, verbose=0)))

    combo = np.asarray(combo)
    combo = np.round(np.mean(combo, 0))
    return combo

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

X_test = data[:10000]
Y_test = targets[:10000]
X_train = data[10000:]
Y_train = targets[10000:]

rewiews_count = 500
X_train = sequence.pad_sequences(X_train, maxlen=rewiews_count)
X_test = sequence.pad_sequences(X_test, maxlen=rewiews_count)
embedding_length = 32

model_lstm = Sequential([
    Embedding(10000, embedding_length, input_length=rewiews_count),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid")])

model_cnn = Sequential([
    Embedding(10000, embedding_length, input_length=rewiews_count),
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPool1D(pool_size=2),
    Dropout(0.4),
    LSTM(128),
    Dropout(0.3),
    Dense(1, activation='sigmoid')])

model_cnl = Sequential([
    Embedding(10000, embedding_length, input_length=rewiews_count),
    Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'),
    MaxPool1D(pool_size=2),
    Dropout(0.2),
    Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPool1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64),
    Dropout(0.1),
    Dense(1, activation='sigmoid')])


train_size = len(X_train) // 3
test_size = len(X_test) // 3
models = [model_lstm, model_cnn, model_cnl]

for i, mod in enumerate(models):
    x_train = X_train[i * train_size : (i + 1) * train_size]
    y_train = Y_train[i * train_size : (i + 1) * train_size]

    x_test = X_test[i * test_size : (i + 1) * test_size]
    y_test = Y_test[i * test_size : (i + 1) * test_size]

    mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    mod.fit(x_train, y_train, validation_split=0.1, epochs=2, batch_size=64)
    scores = mod.evaluate(x_test, y_test, verbose=2)
    print("Accuracy: %.2f%%" % (scores[1]*100))


combo = predict_all(models, X_test, False)
rating = accuracy_score(Y_test, combo)
print("Accuracy: %.2f%%" % (rating*100))



read_text_from_input()  