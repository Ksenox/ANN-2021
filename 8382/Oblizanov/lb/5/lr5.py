from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np


def load_data():
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
    return x_train, y_train, x_test, y_test, num_classes, input_shape, num_train


class CNN:
    def __init__(self):
        self.__set_params()
        self.x_train, self.y_train, self.x_test, self.y_test, self.num_classes, self.input_shape, self.num_train = load_data()

    def __set_params(self):
        self.num_epochs = 20
        self.batch_size = 80
        self.pool_size = (2, 2)
        self.conv_depth_1 = 32
        self.conv_depth_2 = 64
        self.drop_prob_1 = 0.25
        self.drop_prob_2 = 0.5
        self.hidden_size = 512

    def build_model(self, kernel_size=3, dropout=True):
        inp = Input(shape=self.input_shape)
        conv_1 = Convolution2D(self.conv_depth_1, (kernel_size, kernel_size),
                               padding='same', activation='relu')(inp)
        conv_2 = Convolution2D(self.conv_depth_1, (kernel_size, kernel_size),
                               padding='same', activation='relu')(conv_1)
        pool_1 = MaxPooling2D(pool_size=self.pool_size)(conv_2)
        if dropout:
            drop_1 = Dropout(self.drop_prob_1)(pool_1)
        else:
            drop_1 = pool_1
        conv_3 = Convolution2D(self.conv_depth_2, (kernel_size, kernel_size),
                               padding='same', activation='relu')(drop_1)
        conv_4 = Convolution2D(self.conv_depth_2, (kernel_size, kernel_size),
                               padding='same', activation='relu')(conv_3)
        pool_2 = MaxPooling2D(pool_size=self.pool_size)(conv_4)
        if dropout:
            drop_2 = Dropout(self.drop_prob_1)(pool_2)
        else:
            drop_2 = pool_2
        flat = Flatten()(drop_2)
        hidden = Dense(self.hidden_size, activation='relu')(flat)
        if dropout:
            drop_4 = Dropout(self.drop_prob_2)(hidden)
        else:
            drop_4 = hidden
        out = Dense(self.num_classes, activation='softmax')(drop_4)
        model = Model(inp, out)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def plot(self, history, dropout, kernel_size):
        if dropout:
            dropout = ', with dropout'
        else:
            dropout = ', no dropout'
        x = range(1, self.num_epochs + 1)
        plt.plot(x, history.history['loss'])
        plt.plot(x, history.history['val_loss'])
        plt.title('Model loss with kernel size = ' + str(kernel_size) + dropout)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'])
        plt.show()
        plt.plot(x, history.history['acc'])
        plt.plot(x, history.history['val_acc'])
        plt.title('Model accuracy with kernel size = ' + str(kernel_size) + dropout)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'])
        plt.show()

    def launch(self):
        model = self.build_model()
        history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size, epochs=self.num_epochs,
                            verbose=1, validation_data=(self.x_test, self.y_test))
        self.plot(history, True, 3)
        model = self.build_model(dropout=False)
        history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size, epochs=self.num_epochs,
                            verbose=1, validation_data=(self.x_test, self.y_test))
        self.plot(history, False, 3)
        for i in [2, 4, 5]:
            model = self.build_model(i)
            history = model.fit(self.x_train, self.y_train,
                                batch_size=self.batch_size, epochs=self.num_epochs,
                                verbose=1, validation_data=(self.x_test, self.y_test))
            self.plot(history, True, i)


net = CNN()
net.launch()
