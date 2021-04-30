import matplotlib.pyplot as plt
import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb


review1 = ["Having already watched most of the series, I can make a personal conclusion that despite some shortcomings and bloopers that can be safely attributed to the artistry, and not the documentary authenticity of the film, the film adaptation came out very good. This is our first Russian film adaptation of the story about a freak and people."]
review2 = ["A story about two strange characters with whom you probably should have laughed, but it was not funny anywhere. The only funny scene is when Louis de Funes was in the frame, simply because he is funny in himself."]
dmns = 10000


def vectorize(sequences, dimension=dmns):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def build_model():
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dmns)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    data = vectorize(data)
    targets = np.array(targets).astype("float32")

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]

    model = models.Sequential()
    model.add(layers.Dense(50, activation="relu", input_shape=(dmns, )))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    results = model.evaluate(test_x, test_y)
    print(results)
    return model


def test_review():
    index = imdb.get_word_index()
    test_x = []
    words = []
    for line in review2:
        lines = line.translate(str.maketrans('', '', ',.?!:;()')).lower()
    for chars in lines:
        chars = index.get(chars)
    if chars is not None and chars < 10000:
        words.append(chars)
    test_x.append(words)
    test_x = vectorize(test_x)
    model = build_model()
    predict = model.predict(test_x)
    print(predict)


test_review()
