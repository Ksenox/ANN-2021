import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.colors as mclr

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['acc']
val_acc = H.history['val_acc']
epochs = range(1, len(loss) + 1)
#Построение графика ошибки
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#Построение графика точности
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 4:3 - loss: 1.0921 - acc: 0.4963 - val_loss: 1.3684 - val_acc: 0.0000e+00
# 30:3 - loss: 0.2177 - acc: 0.9630 - val_loss: 0.3640 - val_acc: 1.0000
# 30:30:3 - loss: 0.0861 - acc: 0.9778 - val_loss: 0.1073 - val_acc: 1.0000
# 50:50:3 - loss: 0.0926 - acc: 0.9704 - val_loss: 0.1034 - val_acc: 1.0000