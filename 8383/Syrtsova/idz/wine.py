import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow.keras.callbacks
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

#считывание датасетов
white_wine = pd.read_csv('winequality-white.csv', sep=';')
red_wine = pd.read_csv('winequality-red.csv', sep=';')
#добавим метку качества
red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low'
                                                          if value <= 5 else 'medium'
                                                              if value <= 7 else 'high')
white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low'
                                                              if value <= 5 else 'medium'
                                                                  if value <= 7 else 'high')
#объединение датасетов
wines = np.concatenate((red_wine, white_wine), axis=0)
#перемешивание
wines = shuffle(wines)
#разделение на химические характеристики и оценку
data = wines[:, 0:11]
targets = wines[:, 12]
data = np.array(data).astype("float32")
#переход от текстовых меток к категориальному виду
encoder = LabelEncoder()
encoder.fit(targets)
encoded_Y = encoder.transform(targets)
targets = to_categorical(encoded_Y)
#разделение на обучающий и тестовый наборы
train_x, test_x, train_y, test_y = train_test_split(data, targets, test_size=0.2)

res = dict()
los = dict()

def build_models():
    models = []
    model_1 = Sequential()
    model_1.add(Dense(512, activation="relu", input_shape=(11,)))
    model_1.add(Dense(256, activation="relu"))
    model_1.add(Dense(3, activation='softmax'))
    models.append(model_1)

    model_2 = Sequential()
    model_2.add(Dense(256, activation="relu", input_shape=(11,)))
    model_2.add(Dense(3, activation='softmax'))
    models.append(model_2)

    model_3 = Sequential()
    model_3.add(Dense(256, activation="relu", input_shape=(11, )))
    model_3.add(Dense(256, activation="relu"))
    model_3.add(Dropout(0.3, noise_shape=None, seed=None))
    model_3.add(Dense(128, activation="relu"))
    model_3.add(Dense(128, activation="relu"))
    model_3.add(Dropout(0.2, noise_shape=None, seed=None))
    model_3.add(Dense(64, activation="relu"))
    model_3.add(Dense(3, activation="softmax"))
    models.append(model_3)
    return models


models = build_models()
k = 0
for learning_rate in [0.001, 0.01]:
    for optimizers in [Adam(), RMSprop()]:
        for epochs in [50, 100]:
            for model in models:
                optimizer_config = optimizers.get_config()
                model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(train_x, train_y, epochs=epochs, batch_size=64,
                                    validation_data=(test_x, test_y))
                loss, acc = model.evaluate(test_x, test_y)
                print('test_acc:', acc)
                '''
                plt.title('Training and validation accuracy')
                plt.plot(history.history['accuracy'], 'b', label='Training accuracy')
                plt.plot(history.history['val_accuracy'], 'g', label='Validation accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()
                # plt.savefig("%s_%s_%s_%s_acc.png" % (optimizer_config["name"], optimizer_config["learning_rate"], epochs, acc), format='png')
                plt.clf()

                plt.title('Training and validation loss')
                plt.plot(history.history['loss'], 'b', label='Validation loss')
                plt.plot(history.history['val_loss'], 'g', label='Validation loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                # plt.savefig("%s_%s_%s_%s_loss.png" % (optimizer_config["name"], optimizer_config["learning_rate"], epochs, acc), format='png')
                plt.clf()
                '''
                res["%s %s %s %s" % (optimizer_config["name"], optimizer_config["learning_rate"], epochs, acc)] = acc
                los["%s %s %s %s" % (optimizer_config["name"], optimizer_config["learning_rate"], epochs, loss)] = loss
                model.save("model%s.h5" % k)
                k += 1


predictions = 0
print("accuracy:\n")
for i in res.keys():
    print(i, "\n", res[i], "\n")
for i in range(0, k):
    model = load_model("model%i.h5" % i)
    predictions += model.predict(test_x)
predictions = np.greater_equal(predictions, np.array([0.5]))
acc = predictions.mean()
print("Accuracy of ensemble  %s" % acc)
