import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras import layers
from keras.utils import to_categorical
from keras import models
from keras.datasets import imdb

files = ['good1.txt', 'good2.txt', 'bad1.txt', 'bad2.txt']

def vectorize(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def load_text(filename, idim):
    data = []
    print("\n", filename, ":")
    with open(filename, 'r') as file:
        for line in file.readlines():
            print(line)
            data += [w.strip(''.join(['.', ',', ':', ';', '!', '?', '(', ')'])).lower() for w in line.strip().split()]
    index = imdb.get_word_index()
    x_test = []
    for w in data:
        if w in index and index[w] < idim:
            x_test.append(index[w])
    x_test = vectorize([np.array(x_test)], idim)
    return x_test

def predict_success(x_test, model):
    prediction = model.predict(x_test)
    print(prediction)
    if prediction > 0.5 :
      print("This film is going to be successful")
    else:
      print("This film is going to be unsuccessful")

def fit_model(idim):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=idim)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data, idim)
    targets = np.array(targets).astype("float32")

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]

    model = Sequential()
    model.add(layers.Dense(50, activation = "relu", input_shape=(idim, )))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation = "sigmoid"))

    model.compile( optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    history = model.fit(train_x, train_y, epochs = 7, batch_size = 4000, validation_data = (test_x, test_y))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'm', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'm', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    results = model.evaluate(test_x, test_y)
    print(results)
    return model

sizes = [10000]
for size in sizes:
  print("Testing with size", size)
  model = fit_model(size)

print("Testing on custom texts")
good = load_text(files[0], 10000)
predict_success(good, model)

bad = load_text(files[1], 10000)
predict_success(bad, model)

good = load_text(files[2], 10000)
predict_success(good, model)

bad = load_text(files[3], 10000)
predict_success(bad, model)

