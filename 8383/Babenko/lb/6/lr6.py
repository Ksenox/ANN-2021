import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

strings = ["In the film, a demonstration of outright stupidity, humor below the waist, toilet scenes with diarrhea, funny chases for robbers at a funeral. Agitation for easy money, since working is not an option: 'work for slaves'. Open neglect and insult of the older generation, although in the Caucasus it is just the opposite. The theme of the film is, in general, friendship, and the most disgusting thing is how the main character treats his best friend."]


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def prepare_data(dimension):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data, dimension)
    targets = np.array(targets).astype("float32")
    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]
    return (train_x, train_y), (test_x, test_y)


def build_model(dimension):
    model = models.Sequential()
    model.add(layers.Dense(50, activation="relu", input_shape=(dimension,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def model_fit(train_x, train_y, test_x, test_y, dimension):
    model = build_model(dimension)
    H = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))
    draw_plot(H)
    return H


def test_dim(dimension):
    (train_x, train_y), (test_x, test_y) = prepare_data(dimension)
    H = model_fit(train_x, train_y, test_x, test_y, dimension)
    return H.history['val_accuracy'][-1]


def test_dimensions():
    dimensions = [2000, 4000, 6000, 8000, 10000]
    val_accs = []
    for dim in dimensions:
        val_accs.append(test_dim(dim))
    plt.plot(dimensions, val_accs)
    plt.title('Validation accuracy')
    plt.xlabel('Dimensions')
    plt.ylabel('Accuracy')
    plt.show()


def text_load():
    dictionary = dict(imdb.get_word_index())
    test_x = []
    test_y = np.array(answers)
    for string in strings:
        words = string.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('\n', ' ').split()
        num_words = []
        for word in words:
            word = dictionary.get(word)
            if word is not None and word < 10000:
                num_words.append(word)
        test_x.append(num_words)
    test_x = [vectorize(test_x)]
    model = build_model(10000)
    (train_x, train_y), (s1, s2) = prepare_data(10000)
    model.fit(train_x, train_y, epochs=2, batch_size=500)
    predictions = model.predict(test_x)
    print(predictions)


test_dimensions()
text_load()
