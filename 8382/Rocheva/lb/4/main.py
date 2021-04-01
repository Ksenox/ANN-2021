import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from PIL import Image


def compile_model(model, optimizer):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    train_acc = history.history['accuracy'][-1]
    train_loss = history.history['loss'][-1]
    print('test_acc:', test_acc)
    return [train_acc, train_loss, test_acc, test_loss]


# def draw_results(results):
#     x = range(1, len(results) + 1)
#     y = []
#     for i in range(0, len(results)):
#         y.append(results[i][0])
#     plt.plot(x, y, 'bo', label='Train accuracy')
#
#     y = []
#     for i in range(0, len(results)):
#         y.append(results[i][2])
#     plt.plot(x, y, 'ro', label='Test accuracy')
#
#     plt.title('Train and test accuracy')
#     plt.xlabel('Optimizer')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()
#
#     plt.clf()
#
#     y = []
#     for i in range(0, len(results)):
#         y.append(results[i][1])
#     plt.plot(x, y, 'bo', label='Train loss')
#
#     y = []
#     for i in range(0, len(results)):
#         y.append(results[i][3])
#     plt.plot(x, y, 'ro', label='Test loss')
#
#     plt.title('Train and test loss')
#     plt.xlabel('Optimizer')
#     plt.ylabel('loss')
#     plt.legend()
#     plt.show()


def load_img(filename):
    image = Image.open(filename).convert('L')
    image = 255 - np.array(image)
    image = image/255
    return np.expand_dims(image, axis=0)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#
opts = [optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
        optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.9, amsgrad=True),
        optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.9),
        optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.9),
        optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
        optimizers.SGD(learning_rate=0.03, momentum=0.1, nesterov=True),
        optimizers.Adadelta(learning_rate=1.0, rho=0.95),
        optimizers.Adadelta(learning_rate=1.1, rho=0.99),
        optimizers.RMSprop(learning_rate=0.001, rho=0.9),
        optimizers.RMSprop(learning_rate=0.005, rho=0.92)]
#
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# results = []
#
# for opt in opts:
#     results.append(compile_model(model, opt))
# draw_results(results)

compile_model(model, opts[8])

for i in range(1, 10):
    image = load_img('images/' + str(i) + '.png')
    print(i, ':', model.predict_classes(image))
