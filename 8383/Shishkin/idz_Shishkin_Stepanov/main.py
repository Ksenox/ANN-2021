import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow import keras


def clean_punctuation(joke):
    # убирает из строки все, что не соответствует регулярному выражению
    tokens = re.findall(r"[\w']+|[.,!?;]+", joke)
    cleaned = []

    # если в токене хранится несколько знаков препинания подряд, то, например, если там есть вопрос, то в список
    # cleaned добавляется только вопросительный знак. Пример: ["...?"] -> ["?"]
    for token in tokens:
        if '?' in token:
            cleaned.append('?')
        elif '!' in token:
            cleaned.append('!')
        elif '..' in token:
            cleaned.append('...')
        else:
            cleaned.append(token)

    # если предложение не заканчивается на '.', '?', '!', то в конец ставится точка.
    if '.' not in cleaned[-1] and '?' not in cleaned[-1] and '!' not in cleaned[-1]:
        cleaned.append('.')
    return " ".join(cleaned)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds.astype('float64'))
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_overlapping_encoded_sequences(jokes, maxlen, step):
    sentences = []
    next_words = []  # holds the targets
    for joke in jokes:
        for j in range(0, len(joke) - maxlen, step):
            sentences.append(joke[j: j + maxlen])
            next_words.append(joke[j + maxlen])
    return sentences, next_words


# загрузка данных
short_jokes = pd.read_csv('./shortjokes.csv')[:20000]

# т.к. в данных 2 столбца: ID и Joke, мы берем только Joke
jokes = []
for value in short_jokes['Joke']:
    jokes.append(value.lower())

jokes = list(map(clean_punctuation, jokes))
text = ' '.join(jokes)  # преобразование из списка в один текст

tokenizer = Tokenizer(filters='"#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n')  # filters - не учитывает выбранные символы
# создает словарь индексов и слов по популярности (ключ с самым популярным словом имеет значение 1)
tokenizer.fit_on_texts(jokes)
vocab_size = len(tokenizer.word_index) + 1  # количество уникальных слов/символов
print('Vocab Size', vocab_size)

# разбиение шуток в последовательности длиной 11
seq_length = 11
step = 3
integer_encoded_docs = tokenizer.texts_to_sequences(jokes)  # заменяет слова и символы в тексте на значения из словаря
integer_encoded_docs = pad_sequences(integer_encoded_docs,
                                     padding='post')  # добавление 0 к спискам, размер которых меньше 11
split_encoded_docs, next_words = generate_overlapping_encoded_sequences(integer_encoded_docs, seq_length, step)
# размерность padded_docs = (len(split_encoded_docs), 11)
padded_docs = pad_sequences(split_encoded_docs, padding='post')
next_words = np.asarray(next_words)  # нужно получить следующее слово для каждого из этих
print("Number of Sequences:", len(padded_docs))

# Векторизация последовательностей
y = np.zeros((len(padded_docs), vocab_size), dtype=np.bool)
for i, padded_doc in enumerate(padded_docs):
    y[i, next_words[i]] = 1

num_epochs = 20
interval = 1


class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        super(MyCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch == num_epochs - 1 or epoch % interval == 0:
            word_index = tokenizer.word_index  # словарь индексов и слов, где ключи - слова
            index_to_word = dict(
                (index, word) for word, index in word_index.items())  # такой же словарь, где ключи - индексы
            max_words = 5
            maxlen = padded_docs.shape[1]
            print("_________________________________________________________________")
            start_index = random.randint(0, len(text.split(' ')) - max_words - 1)
            generated_text = " ".join(text.split(' ')[start_index: start_index + max_words])
            integer_encoded_gen_text = tokenizer.texts_to_sequences([generated_text])
            readable_gen_text = " ".join(map(lambda key: index_to_word[key], integer_encoded_gen_text[0]))
            print("Random Seed:")
            print(readable_gen_text)

            for _ in range(35):
                integer_encoded_gen_text = tokenizer.texts_to_sequences([generated_text])
                padded_gen_text = pad_sequences(integer_encoded_gen_text, maxlen=maxlen, padding='pre')
                preds = model.predict(padded_gen_text, verbose=0)[0]
                next_index = sample(preds)
                if next_index == 0:
                    break
                most_probable_next_word = index_to_word[next_index]
                print('Generated:', generated_text, 'Next: ', most_probable_next_word)
                generated_text += " " + most_probable_next_word
                readable_gen_text += " " + most_probable_next_word
                generated_text = " ".join(generated_text.split(' ')[1:])
                if most_probable_next_word in ('.', '?', '!'):
                    break

            print('\nFull generated text:')
            print(readable_gen_text)
            print("\n_________________________________________________________________")


filepath = "./weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tb_callback = TensorBoard(log_dir="./logs", histogram_freq=2, write_graph=True, embeddings_freq=1)
callbacks_list = [checkpoint, MyCallback(), tb_callback]

embedding_dim = 256
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=padded_docs.shape[1], mask_zero=True))
model.add(Bidirectional(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(2048, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())
model.fit(padded_docs, y, batch_size=512, epochs=num_epochs, callbacks=callbacks_list)

