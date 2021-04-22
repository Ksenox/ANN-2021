from tensorflow.python.keras.models import Sequential

import var6
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Conv2D
from sklearn.preprocessing import LabelEncoder

train_size = 3000
validation_split = 0.2
test_size = 1000

dataset_image, dataset_labels = var6.gen_data(size=train_size+test_size)

image_shuffle, labels_shuffle = shuffle(dataset_image, dataset_labels)
encoder = LabelEncoder()
labels_shuffle_enc = encoder.fit_transform(labels_shuffle)
labels_shuffle_enc_cat = np_utils.to_categorical(labels_shuffle_enc)

dataset_image, dataset_labels = image_shuffle, labels_shuffle_enc_cat

# разделение данных
train_image = dataset_image[0:train_size]
test_image = dataset_image[train_size:]

train_labels = dataset_labels[0:train_size]
test_labels = dataset_labels[train_size:]

train_image = train_image.reshape(train_image.shape[0], 50, 50, 1)
test_image = test_image.reshape(test_image.shape[0], 50, 50, 1)

# Глобальные параметры
batch_size = 16
num_epochs = 20
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
dense_size_1 = 256
dense_size_2 = 3

height = train_image.shape[1]
width = train_image.shape[2]
depth = 1  # черно - белые изображения

print(train_image.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=(height, width, depth)))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(rate=drop_prob_1))
model.add(Flatten())
model.add(Dense(dense_size_1, activation='relu'))
model.add(Dropout(rate=drop_prob_2))
model.add(Dense(dense_size_2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
hist = model.fit(train_image, train_labels, batch_size=batch_size, epochs=num_epochs,
                 verbose=1, validation_split=0.25)
model.evaluate(test_image, test_labels, verbose=1)


loss = hist.history['loss']
acc = hist.history['accuracy']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(loss) + 1)

# построение графика ошибки
plt.plot(epochs, loss, label='Training loss', linestyle='--', linewidth=2, color="red")
plt.plot(epochs, val_loss, 'b', label='Validation loss', color="blue")
plt.title('Training and Validation loss')                           #оглавление на рисунке
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Построение графика точности
plt.clf()
plt.plot(epochs, acc, label='Training accuracy', linestyle='--', linewidth=2, color="red")
plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color="blue")
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()