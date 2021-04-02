from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
%matplotlib inline

def openImage(path):
    return numpy.asarray(Image.open(path))
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)  #берем первые 25 изображений с пом. ф-ции imshow и отображаем на экране

plt.show()

model = Sequential()
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

def compile_fit_print(optimizer, name):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    H = model.fit(x_train, y_train_cat, epochs=5, batch_size=128, validation_data=(x_test, y_test_cat))

    plt.figure(1, figsize=(8, 5))
    plt.title('Training and test accuracy ' + name)
    plt.plot(H.history['accuracy'], 'r', label='train')
    plt.plot(H.history['val_accuracy'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()

    plt.figure(1, figsize=(8, 5))
    plt.title('Training and test loss ' + name)
    plt.plot(H.history['loss'], 'r', label='train')
    plt.plot(H.history['val_loss'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()


compile_fit_print(optimizers.Adam(), 'adam')

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)), #Первый слой должен преобразовывать изображение 28x28 пикселей в вектор из 784 элементов, 1 - 1 bite of grey(from 255)
    Dense(128, activation='relu'),  #Следующий слой создадим с помощью уже известного нам класса Dense, который свяжет все 784 входа со всеми 128 нейронами.
    Dense(10, activation='softmax') #И такой же последний слой из 10 нейронов, который будет связан со всеми 128 нейронами предыдущего слоя.
])

print(model.summary())

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

n = 0
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"The number is: {np.argmax(res)}") #индекс макс значения (в данном случае это будет 7)

plt.imshow(x_test[n], cmap = plt.cm.binary)
plt.show()

n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"The number is: {np.argmax(res)}") #индекс макс значения (в данном случае это будет 2)

plt.imshow(x_test[n], cmap = plt.cm.binary)
plt.show()

pred = model.predict(x_test)  #пропускаем всю тестовую выборку
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

mask = pred == y_test   #поэлементно сравниваем значения
print(mask[:10])

x_false = x_test[~mask] #выделаем неверные результаты распознавания
p_false = pred[~mask] #их значения

print(x_false.shape)

for i in range(5):
  print("Значение сети: "+str(p_false[i]))
  plt.imshow(x_false[i], cmap=plt.cm.binary)
  plt.show()
