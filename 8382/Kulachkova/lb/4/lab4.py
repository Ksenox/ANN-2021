import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np


# Загрузка изображения и классификация
def load_and_predict(path, model):
    img = load_img(path, color_mode="grayscale", target_size=(28, 28))
    input_arr = img_to_array(img)
    input_arr -= 255
    input_arr = input_arr / -255.0
    plt.imshow(input_arr, cmap=plt.cm.binary)
    plt.show()
    input_arr = np.asarray([input_arr])
    pr = model.predict(input_arr)
    cl = np.argmax(pr, 1)[0]
    return cl
    
    
# Загрузка датасета и преобразование данных
mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Построение модели
model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Тестирование модели
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# Классификация пользовательских изображений
for i in range(10):
    dg = load_and_predict("digits/" + str(i) + ".png", model)
    print(dg)
