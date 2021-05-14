import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D 
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
import matplotlib.pyplot as plt

max_review_length = 500
top_words = 10000
embedding_vector_length = 32

reviews = ["Wow, this film is really beautiful, I like it very much", 
           "What a bad acting, feeling like I'm at a children's party. The premise of the film is empty and uninteresting",
           "I really love the actor who played this role, and in this film he did not disappoint, the picture is very memorable and surprising"]

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
training_data = sequence.pad_sequences(training_data, maxlen=max_review_length)
testing_data = sequence.pad_sequences(testing_data, maxlen=max_review_length)

def lstm_model():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def lstm_dense_model():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dropout(0.4))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def conv_lstm_model():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(LSTM(150))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, name):
    H = model.fit(training_data, training_targets, validation_split=0.1, epochs=2, batch_size=256)
    model.save(name + '.h5')
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss of ' + name + ' model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy of ' + name + ' model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return model

def ensemble(models):
    predict_a = models[0].predict(testing_data)
    predict_b = models[1].predict(testing_data)
    predict_c = models[2].predict(testing_data)
    prediction = numpy.array([1 if (0.3 * predict_a[k] + 0.6 * predict_b[k] + 0.1 * predict_c[k]) > 0.5 else 0 for k in range(len(predict_a))])
    acc = [1 if prediction[i] == testing_targets[i] else 0 for i in range(len(prediction))]
    acc = acc.count(1) / len(acc)
    print(acc)

def test_text(line, models):
    data = [w.strip(''.join(['.', ',', ':', ';', '!', '?', '(', ')'])).lower() for w in line.strip().split()]
    index = imdb.get_word_index()
    x_test = []
    for w in data:
        if w in index and index[w] < top_words:
            x_test.append(index[w] + 3)
    x_test = sequence.pad_sequences([x_test], maxlen=max_review_length)
    predict_a = models[0].predict(x_test)
    predict_b = models[1].predict(x_test)
    predict_c = models[2].predict(x_test)
    return 1 if (0.5 * predict_a + 0.4 * predict_b + 0.1 * predict_c > 0.5) else 0

    
lstm_dense_model = lstm_dense_model()
conv_lstm_model = conv_lstm_model()
lstm_model = lstm_model()
lstm_dense_model = train_model(lstm_dense_model, "LSTM-Dense")
conv_lstm_model = train_model(conv_lstm_model, "Conv-LSTM")
lstm_model = train_model(lstm_model, "LSTM")
ensemble([lstm_dense_model, conv_lstm_model, lstm_model])

for r in reviews:
  res = test_text(r, [lstm_dense_model, conv_lstm_model, lstm_model])
  print("Review:")
  print(r)
  print("Prediction:", res)