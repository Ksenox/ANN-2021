from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adam
from PIL import Image
mnist = tf.keras.datasets.mnist


def get_img(path):
    image_file = Image.open(path)
    image_file = image_file.convert('L')
    resized_image = image_file.resize((28, 28))
    return (np.array(resized_image)) / 255.0


def func(model, path):
    ress = model.predict(np.array([get_img(path)]))
    print(np.argmax(ress))


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99)
# opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
# opt = Adam(learning_rate=0.0001, beta_1=0.99, beta_2=0.99)
# opt = Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
# opt = Nadam(learning_rate=0.001, beta_1=0.99, beta_2=0.999)
opt = Nadam(learning_rate=0.01, beta_1=0.999, beta_2=0.9999)
# opt = RMSprop(learning_rate=0.1, rho=0.9, centered=True)
# opt = RMSprop(learning_rate=0.01, rho=0.9, centered=True)
# opt = RMSprop(learning_rate=0.001, rho=0.9, centered=True)
model.compile(optimizer="nadam", loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels,
                    epochs=5, batch_size=128, verbose=0)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# func(model, "./data/0.png")
# func(model, "./data/1.png")
func(model, "./data/2.png")
# func(model, "./data/3.png")
# func(model, "./data/4.png")
# func(model, "./data/5.png")
# func(model, "./data/6.png")
# func(model, "./data/7.png")
# func(model, "./data/8.png")
# func(model, "./data/9.png")
