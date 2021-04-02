import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.preprocessing import image



def predictNumber(filename):
    img = image.load_img(filename, color_mode="grayscale")
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    img_array = np.array(img.resize((28, 28)))
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict_classes(img_batch)
    print(prediction)
    # I received warning:
    # `model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`
    # So I added two variants and chose first one
    print("Second:")
    predictionSecond = np.argmax(model.predict(img_batch), axis=-1)
    print(predictionSecond)



mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

#plt.imshow(train_images[3],cmap=plt.cm.binary)
#plt.show()
#print(train_labels[3])


train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

#opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.7, momentum=0.0, epsilon=1e-9)
#opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, nesterov=True)
#opt = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-07)
opt = tf.keras.optimizers.Adagrad(learning_rate=0.1, initial_accumulator_value=0.01, epsilon=1e-07)
#opt = tf.keras.optimizers.Adamax(learning_rate=0.01, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
#opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.8, beta_2=0.787, epsilon=1e-08)
#opt = tf.keras.optimizers.Ftrl(learning_rate=0.0001, learning_rate_power=-10.0001, initial_accumulator_value=1.9,
                             #  l1_regularization_strength=19.000001, l2_regularization_strength=0.0000007,
                             #  l2_shrinkage_regularization_strength=8.9, beta=0.6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print('test_acc in %:', test_acc*100)

flag = True
while flag:
    print("Enter a picture name:")
    filename = input()
    if filename == "exit":
        flag = False
    else:
        predictNumber(filename)



