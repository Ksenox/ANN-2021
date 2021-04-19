import var3
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.python.keras import utils
from sklearn.preprocessing import LabelEncoder


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
    out = Dense(2, activation='sigmoid')(drop_3)

    model = Model(inputs=inp, outputs=out) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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

for i in range(K):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], 
                                        train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], 
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=epochs, batch_size=batch_size, validation_split=0, verbose=0)
    val_bc, val_acc = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_acc)

print(np.mean(all_scores))

