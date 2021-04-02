from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizator = optimizers.RMSprop(learning_rate=0.01, rho=0.5)
# optimizator = optimizers.Adam(learning_rate=0.01, beta_1=0.999, beta_2=0.999)
# optimizator = optimizers.Nadam(learning_rate=0.01, beta_1=0.9999, beta_2=0.9999)
# optimizator = optimizers.Adadelta(learning_rate=0.9, rho=0.9)
# optimizator = optimizers.Adamax(learning_rate=0.003, beta_1=0.99, beta_2=0.99)

model.compile(optimizer=optimizator, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss: ', np.around(test_loss, decimals=5))
print('test_acc: ', test_acc)


while True:
    print('Введите имя файла с изображением: ')
    filename = input()
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0
    digit = np.argmax(model.predict(img), axis=-1)

    print(digit[0])
