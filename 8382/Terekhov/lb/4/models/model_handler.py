import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical
from models.model import Model


class ModelHandler:
    def __init__(self, filename: str, params: dict = None):
        self._data_handling()
        self._path = f'models/obj/{filename}'
        if os.path.exists(self._path):
            self._model = load_model(self._path)
            self._history = self._get_history()
        else:
            if params is None:
                raise TypeError("ModelHandler() missing required argument 'params' or correct filename")
            # self._model = Model(self._X_train, self._y_train, self._X_test, self._y_test, **params)
            self._crossval_cycle(**params)
            # self.save()

    def _data_handling(self):
        mnist = tf.keras.datasets.mnist
        (self._X_train, self._y_train), (self._X_test, self._y_test) = mnist.load_data()
        self._X_train = self._X_train / 255.0
        self._X_test = self._X_test / 255.0
        self._y_train = to_categorical(self._y_train)
        self._y_test = to_categorical(self._y_test)

    def save(self):
        self._model.save(self._path)
        path_list = self._path.split('/')
        path = f"{path_list[0]}/{path_list[1]}/histories/{path_list[2].replace('h5', 'npy')}"
        with open(path, 'w'):
            np.save(path, self._history)

    def _get_history(self):
        path_list = self._path.split('/')
        path = f"{path_list[0]}/{path_list[1]}/histories/{path_list[2].replace('h5', 'npy')}"
        return np.load(path, allow_pickle='TRUE')

    def get_history(self):
        return self._history

    def _crossval_cycle(self, **params):
        num_val_samples = len(self._X_train) // 4
        all_scores = []
        all_models = []
        all_histories = []
        for i in range(4):
            print(f"fold {i}")
            val_data = self._X_train[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self._y_train[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                [self._X_train[:i * num_val_samples], self._X_train[(i + 1) * num_val_samples:]],
                axis=0)
            partial_train_targets = np.concatenate(
                [self._y_train[:i * num_val_samples], self._y_train[(i + 1) * num_val_samples:]], axis=0)
            model = Model(params['layers'])
            fitted = model.fit_model(params['optimizer'], partial_train_data, partial_train_targets, val_data,
                                     val_targets)
            val_loss, val_accuracy = model.evaluate(self._X_test, self._y_test)
            all_models.append(model)
            all_scores.append(val_accuracy)
            all_histories.append(pd.DataFrame(fitted.history))
        mean_accuracy = np.mean(all_scores)
        nearest_to_mean = self.find_nearest_idx(np.array(all_scores), mean_accuracy)
        self._model = all_models[nearest_to_mean]
        self._history = all_histories[nearest_to_mean]
        print(all_scores)
        print(mean_accuracy)
        print(nearest_to_mean)
        self.save()

    @staticmethod
    def find_nearest_idx(a, a0):
        idx = np.abs(a - a0).argmin()
        return idx
