import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.python.training.adam import AdamOptimizer
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# load and prepare the image
def load_image(filename):
    # load the image
    loaded_img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    loaded_img = img_to_array(loaded_img)
    # reshape into a single sample with 1 channel
    loaded_img = loaded_img.reshape(1, 28, 28, 1)
    # prepare pixel data
    loaded_img = loaded_img.astype('float32')
    loaded_img = loaded_img / 255.0
    return loaded_img


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()

# цвета от 0 до 255
train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

img = load_image('img/sample_image_1.png')
pred = model.predict(img.reshape(1, 28, 28, 1))
digit = (model.predict(img) > 0.5).astype("int32")
print("First sample image:")
print(pred)
for i in range(len(digit[0])):
    if digit[0][i] == 1:
        print("Your num is " + str(i))
        break

img2 = load_image('img/sample_image_2.png')
pred2 = model.predict(img2.reshape(1, 28, 28, 1))
digit2 = (model.predict(img2) > 0.5).astype("int32")
print("Second sample image:")
print(pred2)
for i in range(len(digit2[0])):
    if digit2[0][i] == 1:
        print("Your num is " + str(i))
        break

img3 = load_image('img/sample_image_3.png')
pred3 = model.predict(img3.reshape(1, 28, 28, 1))
digit3 = (model.predict(img3) > 0.5).astype("int32")
print("Third sample image:")
print(pred3)
for i in range(len(digit3[0])):
    if digit3[0][i] == 1:
        print("Your num is " + str(i))
        break
