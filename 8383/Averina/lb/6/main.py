import re

import numpy as np
from keras import layers
from keras.datasets import imdb
from tensorflow.python.keras.models import Sequential

num_epochs = 2
batch_size = 500
vec_size = 10000


def vectorize(sequences, dimension=vec_size):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def create():

    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=vec_size)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    data = vectorize(data)
    targets = np.array(targets).astype("float32")

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]

    model = Sequential()
    # Input - Layer
    model.add(layers.Dense(50, activation="relu", input_shape=(vec_size,)))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(test_x, test_y))
    print(np.mean(history.history["val_accuracy"]))
    return model


def convert_text(text):
    text = re.sub(r'[^\w\s]', ' ', text).lower().split(' ')
    indexes = imdb.get_word_index()
    text_indexes = []
    for word in text:
        if word in indexes.keys():
            ind = indexes[word]
            if ind < vec_size:
                text_indexes.append(ind + 3)
    return text_indexes
    
    
def read_text_from_file(num):
    with open('./test/test_' + str(num) + '.txt', 'r') as file:
        return file.read()
    

def main():
    model = create()
    answers = [1, 0, 0, 1]
    for i in range(1, 5):
        text = read_text_from_file(i)
        index_text = convert_text(text)
        prediction = model.predict(vectorize(np.asarray([index_text])))
        prediction = round(prediction[0][0])

        if prediction:
            print('It is positive feedback')
        else:
            print('It is negative feedback')
        if prediction == answers[i - 1]:
            print('Answer is true.')
        else:
            print('Answer is false.')
        print('\n')


if __name__ == '__main__':
    main()

