import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers as opt

from tensorflow.keras.layers import Dense, Flatten              #полносвязанный слой
from tensorflow.keras.models import Sequential                  #сеть прямого распространения
from tensorflow.keras.utils import to_categorical

#Построение графика ошибки для обучающих данных
def plot_loss(loss, epochs):
    plt.plot(epochs, loss, label='Training loss', linestyle='--', linewidth=2, color="darkmagenta")
    plt.title('Training loss')                           #оглавление на рисунке
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#Построение графика точности для обучающих данных
def plot_acc(acc, epochs):
    plt.clf()
    plt.plot(epochs, acc, label='Training acc', linestyle='--', linewidth=2, color="lawngreen")
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def build_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #cur_opt = opt.SGD(learning_rate=0.01, nesterov=False, momentum=0.0)
    #cur_opt = opt.Adagrad(learning_rate=0.1)
    #cur_opt = opt.RMSprop(learning_rate=0.001, rho=0.9)
    cur_opt = opt.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=cur_opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#загрузка обучающих и тестовых данных - 4 массива NumPy
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#проверка корректности загрузки - успешно
#plt.imshow(train_images[2], cmap=plt.cm.binary)
#plt.show()
#print(train_labels[2])

#представление изображения
#print(train_images[0].ndim)
#print(train_images[0].shape)
#print(train_images[0])

#нормализация данных
train_images = train_images / 255.0
test_images = test_images / 255.0
#print(train_images[0])

#изменение формата правильных ответов
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#print(train_labels)

model = build_model()
hist = model.fit(train_images, train_labels, epochs=5, batch_size=128)

loss = hist.history['loss']
acc = hist.history['accuracy']
epochs = range(1, len(loss) + 1)

#проверка работы сети на контрольных данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
#plot_loss(loss, epochs)
#plot_acc(acc, epochs)

print("test_loss: ", test_loss, "\n test_acc: ", test_acc)
model.save("ins.h5")