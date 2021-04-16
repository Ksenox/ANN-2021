import var5
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Подготовка данных
data, labels = var5.gen_data()  # генерация данных
sz = data.shape[0]
img_size = data.shape[1]
labels = labels.reshape(sz)
encoder = LabelEncoder()  # кодирование меток
encoder.fit(labels)
labels = encoder.transform(labels).reshape(sz, 1)
data = data.reshape(sz, img_size**2)
data = np.hstack((data, labels))
rng = np.random.default_rng()
rng.shuffle(data)  # перемешивание данных
labels = data[:, -1].reshape(sz, 1)
data = data[:, :data.shape[1]-1].reshape(sz, img_size, img_size, 1)
data /= np.max(data)  # нормализация
tr_sz = int(sz*0.9 // 1)  # разбиение на тренировочное и тестовое множества
train_data = data[:tr_sz, :]
train_labels = labels[:tr_sz, :]
test_data = data[tr_sz:, :]
test_labels = labels[tr_sz:, :]

# Задание параметров
batch_size = 10
num_epochs = 12
kernel_size = 3
pool_size = 2
conv_depth_1 = 16
conv_depth_2 = 32
hidden_size = 100

# Построение модели
inp = Input(shape=(img_size, img_size, 1))
conv_1 = Convolution2D(conv_depth_1, kernel_size, padding='same', activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
conv_2 = Convolution2D(conv_depth_2, kernel_size, padding='same', activation='relu')(pool_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
flat = Flatten()(pool_2)
hidden = Dense(hidden_size, activation='relu')(flat)
out = Dense(1, activation='sigmoid')(hidden)

model = Model(inp, out)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_data, train_labels,
                    batch_size=batch_size, epochs=num_epochs,
                    verbose=1, validation_split=0.1)

# Оценка модели
model.evaluate(test_data, test_labels, verbose=1)

# Построение графика потерь
epochs = np.arange(0, num_epochs, 1)
tr_loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.plot(epochs, tr_loss, color = "blue", label="train_loss")
plt.plot(epochs, val_loss, color = "red", label="val_loss")
plt.legend()
plt.grid()
plt.title("Loss")
plt.show()

# Построение графика точности
tr_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, tr_acc, color = "blue", label="train_acc")
plt.plot(epochs, val_acc, color = "red", label="val_acc")
plt.legend()
plt.grid()
plt.title("Accuracy")
plt.show()

