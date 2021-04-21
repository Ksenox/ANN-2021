from var7 import gen_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Input, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.utils import np_utils

def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def string_to_categorical(Y):
	Y = LabelEncoder().fit_transform(Y)
	Y = to_categorical(Y)
	return Y

def split_data(X, Y):
	test_size = len(X)//5
	X = X.reshape(-1, 50, 50, 1)
	return X[:test_size], Y[:test_size], X[test_size:], Y[test_size:]

X, Y = gen_data()
X, Y = shuffle(X, Y)
Y = string_to_categorical(Y)
X_test, Y_test, X_train, Y_train = split_data(X, Y)

kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

inp = Input(shape=(50, 50, 1))

conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
drop_1 = Dropout(drop_prob_1)(pool_1)

conv_2 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_2 = Dropout(drop_prob_1)(pool_2)

flat = Flatten()(drop_2)
hidden_1 = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(drop_3)
drop_3 = Dropout(drop_prob_2)(hidden_2)
out = Dense(3, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
          batch_size=30, epochs=10,
          verbose=1, validation_split=0.1)
model.evaluate(X_test, Y_test, verbose=1)


# Plotting accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
