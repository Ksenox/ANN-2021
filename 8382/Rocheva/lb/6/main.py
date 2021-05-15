import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=500)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )


def vectorize(sequences, dimension=1000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


custom_x = [
        "It is bad, i hate it",
        "It's too boring and awful"
        "It's amazing, fantastic and exiting",
        "Fine film, i love it too much",
        "Really good, fantastic actors, i like",
]
custom_y = [0., 0., 1., 1., 1.]


def get_text(custom_x, word_index):
    def get_index(a, index):
        new_list = a.split()
        for i, v in enumerate(new_list):
            new_list[i] = index.get(v)
        return new_list
    for i in range(len(custom_x)):
        custom_x[i] = get_index(custom_x[i], word_index)
    return custom_x


custom_x = get_text(custom_x, imdb.get_word_index())
for index_j, i in enumerate(custom_x):
    for index, value in enumerate(i):
        if value is None:
            custom_x[index_j][index] = 0

data = vectorize(data)
targets = np.array(targets).astype("float32")
custom_y = np.asarray(custom_y).astype("float32")

test_x = data[:1000]
test_y = targets[:1000]
train_x = data[1000:]
train_y = targets[1000:]


model = Sequential()
# Input - Layer
model.add(Dense(50, activation="relu", input_shape=(1000,)))
# Hidden - Layers
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(50, activation="linear", kernel_regularizer=regularizers.l2()))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(100, activation="relu", kernel_regularizer=regularizers.l2()))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(50, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=50, epochs=5, verbose=1, validation_data=(test_x, test_y))
H = history.history

plt.figure(1, figsize=(8, 5))
plt.plot(H['loss'], 'g', label='train')
plt.plot(H['val_loss'], 'r', label='validation')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
plt.clf()

plt.plot(H['accuracy'], 'g', label='train')
plt.plot(H['val_accuracy'], 'r', label='validation')
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Apochs')
plt.legend()
plt.show()
plt.clf()

a, acc = model.evaluate(test_x, test_y)
print('The accuracy is ', acc)

custom_x = vectorize(custom_x)

custom_loss, custom_acc = model.evaluate(custom_x, custom_y)
print('custom_acc:', custom_acc)
preds = model.predict(custom_x)
plt.figure(3, figsize=(8,5))
plt.title("Custom dataset predications")
plt.plot(custom_y, 'r', marker='v', label='truth')
plt.plot(preds, 'b', marker='x', label='pred')
plt.legend()
plt.show()
plt.clf()
