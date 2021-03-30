from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


class Model:
    def __init__(self, layers: list):
        self._model = None
        self._build_model(layers)

    def _build_model(self, layers: list):
        self._model = Sequential()
        self._model.add(Flatten())
        for layer in layers:
            self._model.add(Dense(layer['size'], activation=layer['activation']))
        self._model.add(Dense(10, activation='softmax'))

    def fit_model(self, optimizer_, x_train_, y_train_, x_test_, y_test_):
        self._model.compile(optimizer=optimizer_, loss='categorical_crossentropy', metrics=['accuracy'])
        H = self._model.fit(x_train_, y_train_, validation_data=(x_test_, y_test_), epochs=5, batch_size=128, verbose=2)
        return H

    def save(self, fname):
        self._model.save(fname)

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, x_test, y_test):
        return self._model.evaluate(x_test, y_test, batch_size=128, verbose=0)
