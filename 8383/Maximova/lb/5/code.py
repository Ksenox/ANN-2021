from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
import numpy as np

batch_size = 32     # кол-во обуч. образцов, при достижении которого корректируются веса - 1 итерация - 32 объекта
num_epochs = 20      # кол-во эпох на обучение модели - 200 раз повтор

kernel_size = 3     # размер ядра 3х3
pool_size = 2       # размер подвыборки 2х2

conv_depth_1 = 32   # изначальное кол-во ядер
conv_depth_2 = 64   # после первого пула кол-во ядер увеличивается вдвое

drop_prob_1 = 0.25  # 0.25 - вероятность
drop_prob_2 = 0.5   # в полносвязном - 0.5 - вероятность

hidden_size = 512   # кол-во нейронов в полносвязном слое

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# проверка корректности загрузки - успешно
# plt.imshow(x_train[5])
# plt.show()
# print(x_train[5].ndim)
# print(x_train[5].shape) #(32, 32, 3)
# print(x_train[5])

num_train, height, width, depth = x_train.shape     # 50 000 обучающих примеров
num_test = x_test.shape[0]                          # 10 000 тестовых данных
num_classes = np.unique(y_train).shape[0]           # np.unique - находит уникальные элементы массива и возвр в отсорт виде

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# нормализация данных
x_train = x_train / np.max(x_train)  # [0, 1]
x_test = x_test / np.max(x_test)
# print(x_train)

# изменение формата правильных ответов
y_train = np_utils.to_categorical(y_train, num_classes)     # преобразование вектора класса в двоичную матрицу
y_test = np_utils.to_categorical(y_test, num_classes)
print(y_train)
# сеть: conv1[32] -> conv2[32] -> pool1 -> conv3[64] -> conv4[64] -> pool2 -> flatten -> dense1[512] -> dense2[10] ->SM
inp = Input(shape=(height, width, depth))
# conv1[32] -> conv2[32] -> pool (with dropout)
conv_1 = Convolution2D(filters=conv_depth_1, kernel_size=(kernel_size, kernel_size),
                       padding='same', activation='relu')(inp)
conv_2 = Convolution2D(filters=conv_depth_1, kernel_size=(kernel_size, kernel_size),
                       padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(rate=drop_prob_1)(pool_1)

# conv3[64] -> conv4[64] -> pool2 (with dropout)
conv_3 = Convolution2D(filters=conv_depth_2, kernel_size=(kernel_size, kernel_size),
                       padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(filters=conv_depth_2, kernel_size=(kernel_size, kernel_size),
                       padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(rate=drop_prob_1)(pool_2)

# flatten to 1D, apply Dense -> ReLU (with dropout) -> SoftMax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(rate=drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1)
model.evaluate(x_test, y_test, verbose=1)

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








