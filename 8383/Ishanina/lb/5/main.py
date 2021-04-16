from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

batch_size = 64
num_epochs = 15
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_train, depth, height, width = x_train.shape
num_classes = np.unique(y_train).shape[0]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
input_shape = (depth, height, width)
dropout = False
inp = Input(shape=input_shape)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=pool_size)(conv_2)
if dropout:
    drop_1 = Dropout(drop_prob_1)(pool_1)
else:
    drop_1 = pool_1
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=pool_size)(conv_4)
if dropout:
    drop_2 = Dropout(drop_prob_1)(pool_2)
else:
    drop_2 = pool_2
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
if dropout:
    drop_4 = Dropout(drop_prob_2)(hidden)
else:
    drop_4 = hidden
out = Dense(num_classes, activation='softmax')(drop_4)
model = Model(inp, out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=(x_test, y_test))

x = range(1, num_epochs + 1)
plt.plot(x, history.history['loss'])
plt.plot(x, history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()
plt.plot(x, history.history['accuracy'])
plt.plot(x, history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()