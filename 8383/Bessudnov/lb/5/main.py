import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.python.keras import utils

# Глобальные параметры
batch_size = 75
num_epochs = 10
kernel_size = 7         
pool_size = 2           
conv_depth_1 = 32 
conv_depth_2 = 64   
drop_prob_1 = 0.25    
drop_prob_2 = 0.5     
hidden_size = 512  

# Данные для сети
(X_train, y_train), (X_test, y_test) = cifar10.load_data()     
num_train, height, width, depth= X_train.shape    
num_test = X_test.shape[0]                                   
num_classes = np.unique(y_train).shape[0]          
X_train = X_train.astype('float32')                          
X_test = X_test.astype('float32')
X_train /= np.max(X_train)                                      
X_test /= np.max(X_train) 
Y_train = utils.np_utils.to_categorical(y_train, num_classes)       
Y_test = utils.np_utils.to_categorical(y_test, num_classes)         

# Входной слой
inp = Input(shape=(height, width, depth))

# Слои Свертки и пуллинга
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

# Слой Dropout
drop_1 = Dropout(drop_prob_1)(pool_1)

# Еще слои свертки и пуллинга
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)

# Еще слой Dropout
drop_2 = Dropout(drop_prob_1)(pool_2)


flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

H = model.fit(X_train, Y_train,
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1)


model.evaluate(X_test, Y_test, verbose=1)

#Получение ошибки и точности в процессе обучения
loss = H.history['loss']
val_loss = H.history['val_loss']
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
epochs = range(1, len(loss) + 1)

#Построение графика ошибки
plt.subplot(2, 1, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#Построение графика точности
plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()