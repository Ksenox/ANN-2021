import var6
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


(data, labels) = var6.gen_data()


size, height, width = data.shape
batch_size = 32
num_epochs = 20
kernel_size = 4
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512
num_classes = data.shape[0]

data, labels = shuffle(data, labels)
data = data.astype('float32')
data /= np.max(data)
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
encoded_Y = np.reshape(encoded_Y, (size, 1))
encoded_Y = np_utils.to_categorical(encoded_Y, num_classes)


inp = Input(shape=(height, width, 1))
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                       padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                       padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                       padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                       padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)
model = Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(data, encoded_Y, batch_size=batch_size, epochs=num_epochs, validation_split=0.1)

data_test, labels_test = var6.gen_data(size=100)
data_test /= np.max(data_test)
encoder = LabelEncoder()
encoder.fit(labels_test)
encoded_Y = encoder.transform(labels_test)
encoded_Y = np.reshape(encoded_Y, (100, 1))
encoded_Y = np_utils.to_categorical(encoded_Y, num_classes)

results = model.evaluate(data_test, encoded_Y, verbose=1)
print(results)


model_json = model.to_json()
with open("model_5.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_5.h5")
