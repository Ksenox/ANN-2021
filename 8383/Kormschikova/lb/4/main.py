import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from PIL import Image
import numpy as np
from pathlib import Path


def load_img(path):
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((28, 28))
    return 1 - np.array(img) / 255.0


mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
#
# plt.imshow(train_images[0],cmap=plt.cm.binary)
# plt.show()
# print(test_images[0].shape)
# print(test_images[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(amsgrad=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

path = ""
while(1):
    print("enter path to image or 1 to end")
    path = input()
    if(path == "1"):
        break
    path = Path(path)
    if (path.exists() == False):
        print("File not exist!")
        continue
    img = load_img(path)
    predict = model.predict(np.array([img]))
    print("It is ",np.argmax(predict))
    # plt.imshow(img, cmap=plt.cm.binary)
    # plt.show()

