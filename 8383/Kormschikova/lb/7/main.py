import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
from tensorflow.python.keras.models import load_model

def textFromFile(fileName, dimension=10000):
    file = open(fileName, "r")
    text = file.read()
    file.close()
    text.lower()
    tt = str.maketrans(dict.fromkeys("!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"))
    text = text.translate(tt).split()
    index = imdb.get_word_index()
    codedText = []
    for word in text:
        i = index.get(word)
        if i is not None and i < dimension:
            codedText.append(i+3)
    codedText = np.array([codedText])
    codedText = sequence.pad_sequences(codedText, maxlen=max_review_length)
    return codedText


def createModel(type = 1):
    if(type == 1):
        print("First type")
        model = Sequential()
        model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
        model.add(LSTM(100, dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        print("Second type")
        model = Sequential()
        model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def createEnsemble(count = 5):
    for i in range(count):
        print("Creating model "+ str(i)+"...")
        X_batch_train = X_train[i * train_batch:(i + 1) * train_batch]
        X_batch_test = X_test[i * test_batch:(i + 1) * test_batch]
        Y_batch_train = Y_train[i * train_batch:(i + 1) * train_batch]
        Y_batch_test = Y_test[i * test_batch:(i + 1) * test_batch]
        model = createModel(i % 2)
        # print(model.summary())
        model.fit(X_batch_train, Y_batch_train, validation_data=(X_batch_test, Y_batch_test), epochs=2, batch_size=64)
        score = model.evaluate(X_test, Y_test, verbose=0)
        scores.append(score[1])
        print("Accuracy:", score, score[1], "\n")
        model.save("model_" + str(i) + ".h5")
    print("Mean accuracy:", np.mean(scores))


top_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

max_review_length = 500
train_len = len(data) * 8 // 10
X_train = data[:train_len]
X_test = data[train_len:]
Y_train = targets[:train_len]
Y_test = targets[train_len:]

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

count_model = 5;
scores = []
train_batch = train_len//count_model
test_batch = (len(data) - train_len)//count_model
# print(train_batch, test_batch)
embedding_vector_length = 32


print("Type 1 to create models, another input to load models")
inp = input()
if(inp == "1"):
    createEnsemble(count_model)

models = []
score_load = []
for i in range(count_model):
    print("Loading model_"+str(i)+".h5...")
    model = load_model("model_"+str(i)+".h5")
    models.append(model)
#     score = model.evaluate(X_test, Y_test, verbose=0)
#     score_load.append(score[1])
#     print("Load success. Accuracy:", score[1])
#
# print("Mean accuracy:", np.mean(score_load))


while(1):
    print("Print file name *.txt or stop")
    fileName = input()
    if fileName == "stop":
        break
    codedText = textFromFile(fileName, top_words)
    res = []
    for i in range(count_model):
        r = models[i].predict(codedText)
        print("Res model "+str(i))
        print(r)
        res.append(r)
    result = np.mean(res)
    print("\nAnswer:")
    if result >= 0.5:
        print("Good", result)
    else:
        print("Bad", result)
