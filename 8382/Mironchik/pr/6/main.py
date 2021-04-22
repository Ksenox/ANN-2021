import var6
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras import utils, layers, models
import matplotlib.pyplot as plot


def plot_single_history(history, color="blue"):
    keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
    titles = ["Loss", "Accuracy", "Val loss", "Val accuracy"]
    xlabels = ["epoch", "epoch", "epoch", "epoch"]
    ylabels = ["loss", "accuracy", "loss", "accuracy"]
    #ylims = [3, 1.1, 3, 1.1]

    for i in range(len(keys)):
        plot.subplot(2, 2, i + 1)
        plot.title(titles[i])
        plot.xlabel(xlabels[i])
        plot.ylabel(ylabels[i])
        #plot.gca().set_ylim([0, ylims[i]])
        plot.grid()
        values = history[keys[i]]
        plot.plot(range(1, len(values) + 1), values, color=color)


def gen_prepared_data(size = 500, img_size = 50):
    _data, _labels = var6.gen_data(size=size, img_size=img_size)
    _data = _data.reshape(size, img_size, img_size)
    _labels = _labels.reshape(size)

    indices = np.array(range(size))
    np.random.shuffle(indices)
    _data = np.take(_data, indices, axis=0)
    _labels = np.take(_labels, indices)

    _encoder = LabelEncoder()
    _encoder.fit(_labels)
    _labels = _encoder.transform(_labels).reshape(size)
    _classes = np.unique(_labels).shape[0]
    _labels = utils.to_categorical(_labels, _classes)

    return _data, _labels


size = 1000
test_size = 50
img_size = 50
data, labels = gen_prepared_data(size, img_size)
test_data, test_labels = gen_prepared_data(test_size, img_size)

batch_size = 10
num_epochs = 10
kernel_size = 8
pool_size = 3
conv_depth_1 = 8
conv_depth_2 = 16
conv_depth_3 = 32
hidden_size = 100

inp = layers.Input(shape=(img_size, img_size, 1))
conv_1 = layers.Convolution2D(conv_depth_1, kernel_size, padding='same', activation='relu')(inp)
pool_1 = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
conv_2 = layers.Convolution2D(conv_depth_2, kernel_size, padding='same', activation='relu')(pool_1)
pool_2 = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
conv_3 = layers.Convolution2D(conv_depth_3, kernel_size, padding='same', activation='relu')(pool_2)
pool_3 = layers.MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
flat = layers.Flatten()(pool_3)
hidden = layers.Dense(hidden_size, activation='relu')(flat)
out = layers.Dense(labels.shape[1], activation='softmax')(hidden)

model = models.Model(inp, out)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(data, labels,
                    batch_size=batch_size, epochs=num_epochs,
                    verbose=1, validation_split=0.1)


model.evaluate(test_data, test_labels, verbose=1)
plot_single_history(history.history)
plot.show()
