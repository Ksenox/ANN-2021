import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import load_img, img_to_array

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# for i in range(100):
#     plt.imshow(train_images[i],cmap=plt.cm.binary)
#     plt.show()
#     print(train_labels[i])

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

for i in range(1, 4):
    pic = load_img("./digits/" + str(i) + ".png", color_mode='grayscale', target_size=(28, 28))
    pic_arr = img_to_array(pic)
    pic_arr -= 255
    pic_arr = pic_arr / -255.0
    plt.imshow(pic_arr, cmap=plt.cm.binary)
    plt.show()
    pic_arr = np.asarray([pic_arr])
    predict = model.predict(pic_arr)
    val = np.argmax(predict, 1)[0]
    print(val)


model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

for i in range(1, 4):
    pic = load_img("./digits/" + str(i) + ".png", color_mode='grayscale', target_size=(28, 28))
    pic_arr = img_to_array(pic)
    pic_arr -= 255
    pic_arr = pic_arr / -255.0
    plt.imshow(pic_arr, cmap=plt.cm.binary)
    plt.show()
    pic_arr = np.asarray([pic_arr])
    predict = model.predict(pic_arr)
    val = np.argmax(predict, 1)[0]
    print(val)

    
model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='nadam',loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

for i in range(1, 4):
    pic = load_img("./digits/" + str(i) + ".png", color_mode='grayscale', target_size=(28, 28))
    pic_arr = img_to_array(pic)
    pic_arr -= 255
    pic_arr = pic_arr / -255.0
    plt.imshow(pic_arr, cmap=plt.cm.binary)
    plt.show()
    pic_arr = np.asarray([pic_arr])
    predict = model.predict(pic_arr)
    val = np.argmax(predict, 1)[0]
    print(val)


model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

for i in range(1, 4):
    pic = load_img("./digits/" + str(i) + ".png", color_mode='grayscale', target_size=(28, 28))
    pic_arr = img_to_array(pic)
    pic_arr -= 255
    pic_arr = pic_arr / -255.0
    plt.imshow(pic_arr, cmap=plt.cm.binary)
    plt.show()
    pic_arr = np.asarray([pic_arr])
    predict = model.predict(pic_arr)
    val = np.argmax(predict, 1)[0]
    print(val)
