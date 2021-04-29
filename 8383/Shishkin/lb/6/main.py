import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence
from tensorflow.python.keras.models import Sequential


# VECTORIZE as one cannot feed integers into a NN
# Encoding the integer sequences into a binary matrix - one hot encoder basically
# From integers representing words, at various lengths - to a normalized one hot encoded tensor (matrix) of 10k columns
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# слова заменены целыми числами, которые указывают на абсолютную популярность слова в наборе данных
max_words = 10000
# нас интересуют только первые 10000 самых употребляемых слов в наборе данных.
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=max_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

data = vectorize_sequences(data)
# VECTORIZE the labels too - NO INTEGERS only floats into a tensor...(rare exceptions)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = Sequential()
# Input - Layer
model.add(layers.Dense(50, activation="relu", input_shape=(10000,)))

# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))

# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

NumEpochs = 2
BatchSize = 512

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_x, train_y, epochs=NumEpochs, batch_size=BatchSize, validation_data=(test_x, test_y))
results = model.evaluate(test_x, test_y)
print("_" * 100)
print("Test Loss and Accuracy")
print("results ", results)

while True:
    print("Введите текст (для выхода - exit):")
    text = str(input())
    if text == "exit":
        break
    else:
        words = text_to_word_sequence(text)
        vocabulary = imdb.get_word_index()
        x_predict = []
        for i in range(len(words)):
            if words[i] not in vocabulary:
                continue
            if vocabulary[words[i]]+3 < 10000:
                x_predict.append(vocabulary[words[i]]+3)
        x_predict = vectorize_sequences(np.asarray([x_predict]))
        prediction = (model.predict(x_predict) > 0.5).astype("int32")
        print("Good" if prediction[0] == 1 else "Bad")

# VALIDATION ACCURACY curves
plt.clf()
history_dict = history.history
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, (len(history_dict['accuracy']) + 1))
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# VALIDATION LOSS curves
plt.clf()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history_dict['loss']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
