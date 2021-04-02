
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from keras import optimizers

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

opt = optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.95)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)

with open("tests.txt", "a") as tests_file:
    line = "\n\n" + str(opt.get_config()) + "\n" + "Accuracy: " + str(test_acc)
    tests_file.write(line)


print("test_acc:", test_acc)

print(opt.get_config())

model.save("model_lb4.h5")