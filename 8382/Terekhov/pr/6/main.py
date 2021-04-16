from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from var4 import gen_data

if __name__ == '__main__':
    X, y = gen_data()
    label_encoder = LabelEncoder()
    label_encoder.fit(["Cross", "Line"])
    y = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    batch_size = 32
    num_epochs = 16
    kernel_size = 3
    pool_size = 2
    conv_depth_1 = 32
    conv_depth_2 = 64
    drop_prob_1 = 0.25
    drop_prob_2 = 0.5
    hidden_size = 512

    inputs = Input(shape=(*X.shape[1:], 1))

    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inputs)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size), padding='same')(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(1, activation='sigmoid')(drop_3)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    H = model.fit(X_train, y_train,
                  batch_size=batch_size, epochs=num_epochs,
                  verbose=1, validation_split=0.1)
    model.evaluate(X_test, y_test, verbose=1)
