import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

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

optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9)

model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss: ', test_loss)
print('test_acc: ', test_acc)


def predict_image(model, filename):
    image = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    image = img_to_array(image).reshape(1, 28, 28, 1)
    image = image.astype('float32') / 255.0
    plt.imshow(image[0], cmap=plt.cm.binary)
    plt.show()
    prediction = model.predict(image)
    print(f'File: {filename} - prediction: {np.argmax(prediction)}')


predict_image(model, 'number_1.png')
predict_image(model, 'number_2.png')
predict_image(model, 'number_3.png')
predict_image(model, 'number_5.png')
predict_image(model, 'number_6.png')
