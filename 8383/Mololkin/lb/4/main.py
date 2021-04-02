import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras.optimizers as opts
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten


def input_img(path):
    img = Image.open(path).convert('L')
    matr_img = np.array(img) / 255.0
    matr_img = np.array([matr_img])
    return matr_img


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

#opt = opts.Adam(learning_rate=0.001)
opt = opts.RMSprop(learning_rate=0.001, rho=0.9)
#opt = opts.Nadam(learning_rate=0.0001)

model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=256, verbose=False)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

for i in range(0, 10):
    img_arr = input_img('./images/' + str(i) + '.png')
    result = model.predict(img_arr)
    print(np.argmax(result))
