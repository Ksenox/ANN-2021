import var3
import numpy as np
import matplotlib.pyplot as plt
import pandas
from keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.python.keras import utils
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


target_names = ("Horizontal", "Vertical")
fold_number = 0
epochs_to_watch = []
accuracy_log = []
val_accuracy_log = []

def saveHist(epoch, stats):
    plt.clf()
    data = dict(zip(target_names, stats))
    df = pandas.DataFrame(data, index=[0])
    titleString = "Histogram for " + str(epoch) + " epoch in " + str(fold_number) + " fold"
    yticks = np.arange(0, 1, 0.1)
    df.plot(kind="bar", title=titleString, ylim=[0, 1], yticks=yticks, ylabel="Binary Accuracy")
    plt.savefig("hist_" + str(fold_number) + "_" + str(epoch) + ".png")


class HistogramLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epochs_to_watch.count(epoch) > 0:
            stats = []
            for name in target_names:
                stats.append(logs[name + "_" + "binary_accuracy"])
            saveHist(epoch, stats)

           

def build_model():
    inp = Input(shape=(image_size, image_size, 1))

    # Слои Свертки и пуллинга
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

    # Слой Dropout
    drop_1 = Dropout(drop_conv)(pool_1)

    # Еще слои свертки и пуллинга
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)

    # Еще слой Dropout
    drop_2 = Dropout(drop_conv)(pool_2)


    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_dense)(hidden)

    out = [Dense(2, name=name)(drop_3) for name in target_names]

    model = Model(inputs=inp, outputs=out) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


# Гипер параметры
data_size = 1000
image_size = 50

K = 4
batch_size = 10
epochs = 40
kernel_size = 3
pool_size = 2
drop_conv = 0.1
drop_dense = 0.5
conv_depth_1 = 32 
conv_depth_2 = 64
hidden_size = 32

(data, label) = var3.gen_data(size=data_size, img_size=image_size)

encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label) 
label = utils.np_utils.to_categorical(label, 2)

# Перемещивание данных
rand_index = np.random.permutation(len(label))
train_data = data[rand_index]
train_targets = label[rand_index]

num_val_samples = len(train_data) // K
all_scores = []


path = Path("hists.txt")
if (not path.exists()):
    print("Oops! Can't find file hists.txt")
else:
    epochs_to_watch = ""
    with open(path.absolute()) as f:
        epochs_to_watch = f.read()

    epochs_to_watch = list(map(int, epochs_to_watch.split()))

for i in range(K):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], 
                                        train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], 
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()

    history = model.fit(partial_train_data, partial_train_targets, epochs=epochs, batch_size=batch_size, 
        validation_split=0, verbose=0, callbacks=[HistogramLogger()])

    fold_number += 1
