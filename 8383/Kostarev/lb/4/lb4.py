import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


def load_image(path):
    image = load_img(path, color_mode='grayscale', target_size=(28, 28))
    image_array = img_to_array(image)
    image_array -= 255
    image_array = image_array / -255.0
    plt.imshow(image_array, cmap=plt.cm.binary)
    plt.show()
    image_array = np.asarray([image_array])
    return image_array


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
op_1 = tf.keras.optimizers.Adam(beta_2=0.8)
op_2 = tf.keras.optimizers.RMSprop(rho=0.99)
op_3 = tf.keras.optimizers.Adagrad(learning_rate=0.3)
model.compile(optimizer=op_3, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

for i in range(10):
    arr = load_image('images/' + str(i) + '.png')
    predict = model.predict(arr)
    print(np.argmax(predict, 1)[0])
