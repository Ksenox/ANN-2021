import sklearn.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from var2 import gen_data

data, labels = gen_data()
data, labels = sklearn.utils.shuffle(data, labels)

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

nums, width, height = data.shape

count = nums / 10
data_train = data[count:]
data_test = data[:count]
labels_train = labels[count:]
labels_test = labels[:count]
data_train = data_train.reshape(data_train.shape[0], width, height, 1)
data_test = data_test.reshape(data_test.shape[0], width, height, 1)

batch_size = 10
num_epochs = 20
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 64

model = Sequential()
model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu', input_shape=(width, height, 1)))
model.add(MaxPooling2D((pool_size, pool_size)))
model.add(Dropout(drop_prob_1))
model.add(Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu'))
model.add(MaxPooling2D((pool_size, pool_size)))
model.add(Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu'))
model.add(Dropout(drop_prob_2))
model.add(Flatten())
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(data_train, labels_train, epochs=num_epochs, batch_size=batch_size, validation_data=(data_test, labels_test))


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()
