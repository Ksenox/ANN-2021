import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.datasets import imdb
from keras import Sequential

k = 10000
feedback = ["'Big Rzhaka' is a disgusting film, everything is terrible in it, from the actors to the humor. In general, the movie turned out to be mediocre and soulless. I didn't even think that our 'directors' know how to shoot such a bad and worthless movie. Bottom line: worst movie I've seen, terrible acting, bad script, and disgusting humor."]


def vectorize(sequences, dimension=k):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(
            sequences):
        results[i, sequence] = 1
    return results


def test():
    index = imdb.get_word_index()
    test_x = []
    words = []
    for line in feedback:
        lines = line.translate(str.maketrans('', '', ',.?!:;()')).lower()
    for word in lines:
        word = index.get(word)
        if word is not None and word < 10000:
            words.append(word)
    test_x.append(words)
    test_x = vectorize(test_x)
    model = build_model()
    predict = model.predict(test_x)
    print(predict)


def build_model():
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=k)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    data = vectorize(data)
    targets = np.array(targets).astype("float32")

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]

    model = Sequential()
    model.add(layers.Dense(50, activation="relu", input_shape=(k,)))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    H = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

    loss = H.history['loss']
    val_loss = H.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'm', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    plt.plot(epochs, acc, 'm', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    results = model.evaluate(test_x, test_y)
    print(results)
    return model

#build_model()
test()

