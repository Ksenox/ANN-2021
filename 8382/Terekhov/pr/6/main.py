import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import plot_model

from var4 import gen_data


def plot(data: pd.DataFrame, label: str):
    axis = sns.lineplot(data=data, dashes=False)
    axis.set(ylabel=label, xlabel='epochs', title=label)
    axis.grid(True, linestyle="--")
    plt.show()

if __name__ == '__main__':
    X, y = gen_data()
    label_encoder = LabelEncoder()
    label_encoder.fit(np.unique(y))
    y_encoded = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33)

    batch_size = 64
    num_epochs = 10
    kernel_size = 7
    pool_size = 5
    conv_depth_1 = 32
    drop_prob = 0.45
    hidden_size = 64

    inputs = Input(shape=(*X.shape[1:], 1))

    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inputs)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob)(pool_1)
    flat = Flatten()(drop_1)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_2 = Dropout(drop_prob)(hidden)
    out = Dense(1, activation='sigmoid')(drop_2)

    model = Model(inputs=inputs, outputs=out)
    plot_model(model, show_layer_names=False)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    H = model.fit(X_train, y_train,
                  batch_size=batch_size, epochs=num_epochs,
                  verbose=1, validation_split=0.1)
    model.evaluate(X_test, y_test, verbose=1)
    history = pd.DataFrame(H.history)
    plot(history[['loss', 'val_loss']], "Loss")
    plot(history[['accuracy', 'val_accuracy']], "Accuracy")

