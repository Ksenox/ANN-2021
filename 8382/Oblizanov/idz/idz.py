import string
import datetime
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def read_text(filename):
    with open(filename, mode='rt', encoding='utf-8') as file:
        text = file.read()
        phrases = text.strip().split('\n')
        return [phrase.split('\t') for phrase in phrases]


print("Loading file...")
data = read_text("vocab.txt")
vocab = np.array(data)

print("Dictionary size:", vocab.shape)

print("Formatting text...")
vocab[:, 0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in vocab[:, 0]]
vocab[:, 1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in vocab[:, 1]]

for i in range(len(vocab)):
    vocab[i, 0] = vocab[i, 0].lower()
    vocab[i, 1] = vocab[i, 1].lower()

print("Tokenize...")
data_tokenizer = Tokenizer()
data_tokenizer.fit_on_texts(vocab[:, 0])
data_vsize = len(data_tokenizer.word_index) + 1
data_tsize = 8

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(vocab[:, 1])
label_vsize = len(label_tokenizer.word_index) + 1
label_tsize = 8


def encode_sequences(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq


def get_word(n, tokenizer):
    if n == 0:
        return ""
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return ""


def translate(model, custom_text, custom_answers, epoch=-1):
    custom_data = encode_sequences(data_tokenizer, data_tsize, custom_text)
    prediction = model.predict_classes(custom_data)
    if (epoch > 0):
        print("Epoch" + str(epoch))
    for j, pred in enumerate(prediction):
        print("Original:")
        print(custom_text[j])
        print("Google translate:")
        print(custom_answers[j])
        print("Prediction:")
        output = ""
        for i in range(len(pred)):
            if pred[i] != 0:
                output += str(get_word(pred[i], label_tokenizer)) + " "
            else:
                break
        print(output)
        print()


custom_text = ["i don't want to use this tool", "do you know how to get there",
               "this film is terrible", "i will borrow money to pay", "life is so hard"]
custom_answers = ["Ich möchte dieses Tool nicht verwende", "Weißt du, wie man dort hinkommt",
                  "Dieser Film ist schrecklich", "Ich werde Geld leihen um zu bezahlen", "das Leben ist so hart"]

train, test = train_test_split(vocab, test_size=0.2, random_state=12)

trainX = encode_sequences(data_tokenizer, data_tsize, train[:, 0])
trainY = encode_sequences(label_tokenizer, label_tsize, train[:, 1])

testX = encode_sequences(data_tokenizer, data_tsize, test[:, 0])
testY = encode_sequences(label_tokenizer, label_tsize, test[:, 1])


def make_model(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(n))
    model.add(Dropout(0.3))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(out_vocab, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


class TranslateEveryFive(keras.callbacks.Callback):
    def __init__(self):
        super(TranslateEveryFive, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            translate(self.model, custom_text, custom_answers, epoch + 1)


print("Data vocab size / token size:", data_vsize, data_tsize)
print("Label vocab size / token size:", label_vsize, label_tsize)

print("Input 0 if you want to load model or anything to fit model")
s = input()
if s is not "0":
    print("Initializing model...")
    model = make_model(data_vsize, label_vsize, data_tsize, label_tsize, 512)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    num_epochs = 30
    callbacks_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
        ),
        ModelCheckpoint(
            filepath='vocab-model.h5',
            monitor='val_loss',
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=2,
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1),
        TranslateEveryFive()
    ]

    print("Training...")
    history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                        epochs=num_epochs, batch_size=256,
                        validation_split=0.2, callbacks=callbacks_list, verbose=1)
    model.evaluate(testX, testY.reshape(testY.shape[0], testY.shape[1], 1))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'validation'])
    plt.show()

model = load_model('vocab-model.h5')
translate(model, custom_text, custom_answers)
