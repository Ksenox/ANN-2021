import numpy as np
import pandas as pd
from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import var4


if __name__ == '__main__':
    x, y = var4.gen_data()
    label_encoder = LabelEncoder()
    label_encoder.fit(np.unique(y))
    y_encoded = label_encoder.transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.25)

    batch_size = 10
    num_epochs = 8
    kernel_size = 7
    pool_size = 5
    conv_depth = 32
    drop_prob = 0.4
    hidden_size = 128

    inputs = Input(shape=(*x.shape[1:], 1))

    conv_1 = Convolution2D(conv_depth, (kernel_size, kernel_size), padding='same', activation='relu')(inputs)
    conv_2 = Convolution2D(conv_depth, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    flat = Flatten()(pool_1)
    hidden = Dense(hidden_size, activation='relu')(flat)
    out = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    H = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1)
    model.evaluate(x_test, y_test)
    history = pd.DataFrame(H.history)

