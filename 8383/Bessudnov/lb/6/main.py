import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

vector_size = 10000

def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


if __name__ == "__main__":
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=vector_size)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)



    data = vectorize(data, vector_size)
    targets = np.array(targets).astype("float32")

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]


    model = models.Sequential()
    # Input - Layer
    model.add(layers.Dense(128, activation = "relu", input_shape=(vector_size, )))

    # Hidden - Layers
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(128, activation = "sigmoid"))
    model.add(layers.Dense(256, activation = "relu"))
    model.add(layers.Dropout(0.35, noise_shape=None, seed=None))
    model.add(layers.Dense(128, activation = "relu"))

    # Output- Layer
    model.add(layers.Dense(1, activation = "sigmoid"))
    model.summary()

    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    H = model.fit(train_x, train_y, epochs= 2, batch_size = 750, validation_data = (test_x, test_y), verbose=0)

    print(np.mean(H.history["val_accuracy"]))

    # model.save("model_lb6.h5")
