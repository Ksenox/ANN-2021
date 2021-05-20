import var6
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tensorflow.keras.callbacks import Callback

class CallbackVar1(Callback):  # создали подкласс Callback

    def __init__(self, prefix):
        date = datetime.now()
        self.file_name = '{}-{}-{}_{}_'.format(date.day, date.month, date.year, prefix)

        self.elem = 3
        self.val_acc = [0] * self.elem
        self.rewriting = 0
        self.param = 'val_accuracy'

    def on_epoch_end(self, epoch, logs=None):
        for i in range(self.elem):
            acc_epoch = logs[self.param]
            if self.val_acc[i] < acc_epoch:
                self.val_acc.insert(i, acc_epoch)
                self.model.save(self.file_name + str(i + 1) + '.hdf5')
                if self.rewriting != self.elem - 1:
                    self.rewriting = self.rewriting + 1
                else:
                    self.rewriting = 0
                break

            elif self.rewriting > i:
                continue

    def on_train_end(self, logs=None):
        print("Тройка наилучших моделей:")
        for i in range(self.elem):
            print(self.file_name + str(i + 1) + '.hdf5:' + " val_accuracy = " + str(self.val_acc[i]))

def edit_data(image, labels):
    image_shuffle, labels_shuffle = shuffle(image, labels)                    # 1) перемешивание данных
    encoder = LabelEncoder()
    labels_shuffle_enc = encoder.fit_transform(labels_shuffle)            # 2) преобразовать строки в числа и вернуть рез-т
    labels_shuffle_enc_cat = np_utils.to_categorical(labels_shuffle_enc)  # 3) изменение формата правильных ответов
    return image_shuffle, labels_shuffle_enc_cat

print("Введите Префикс")
prefix = input()

train_size = 3000
validation_split = 0.2
test_size = 1000

dataset_image, dataset_labels = var6.gen_data(size=train_size+test_size)
dataset_image, dataset_labels = edit_data(dataset_image, dataset_labels)

# print(dataset_image[100])
# plt.imshow(dataset_image[200])
# plt.show()

# разделение данных
train_image = dataset_image[0:train_size]
test_image = dataset_image[train_size:]

train_labels = dataset_labels[0:train_size]
test_labels = dataset_labels[train_size:]

# гиперпараметры
batch_size = 16
num_epochs = 15

kernel_size = 3  # размер ядра 3х3
pool_size = 2

conv_depth_1 = 32  # кол-во ядер
conv_depth_2 = 64

drop_prob_1 = 0.25
drop_prob_2 = 0.5

dense_size_1 = 256  # кол-во нейронов в полсносвязном слое
dense_size_2 = 3

height = train_image.shape[1]
width = train_image.shape[2]
depth = 1  # черно - белые изображения

# сеть: conv1[32] -> pool1 -> drop1 -> conv2[32]  -> pool2 -> drop2 -> flatten -> dense1[256] -> drop3 -> dense2[3]
inp = Input(shape=(height, width, depth))
conv_1 = Convolution2D(filters=conv_depth_1, kernel_size=(kernel_size, kernel_size),
                       padding='same', activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
drop_1 = Dropout(rate=drop_prob_1)(pool_1)


conv_2 = Convolution2D(filters=conv_depth_2, kernel_size=(kernel_size, kernel_size),
                       padding='same', activation='relu')(drop_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_2 = Dropout(rate=drop_prob_1)(pool_2)

flat = Flatten()(drop_2)
dense_1 = Dense(dense_size_1, activation='relu')(flat)
drop_3 = Dropout(rate=drop_prob_2)(dense_1)
out = Dense(dense_size_2, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
hist = model.fit(train_image, train_labels, batch_size=batch_size, epochs=num_epochs,
                 verbose=1, validation_split=0.25,
                 callbacks=[CallbackVar1(prefix)])
# передаем список обратных вызовов в fit через параметр callbacks

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

#Построение графика точности
plt.clf()
plt.plot(epochs, acc, label='Training accuracy', linestyle='--', linewidth=2, color="red")
plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color="blue")
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()