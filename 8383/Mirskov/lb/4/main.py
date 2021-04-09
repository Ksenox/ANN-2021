import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
import numpy as np

def predict_user_image():
    path = './image2.png'

    img = tf.keras.preprocessing.image.load_img(path, color_mode = 'grayscale', target_size = (28,28))
    input_arr = np.array([tf.keras.preprocessing.image.img_to_array(img)]) / 255.0

    prediction_result = model.predict(input_arr)
    return np.argmax(prediction_result, 1)[0]

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# plt.imshow(train_images[0],cmap=plt.cm.binary)
# plt.show()
# print(train_labels[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
#optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1, momentum = 0.1)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.1)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001, rho = 0.09)

model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

print("user image:", predict_user_image())

