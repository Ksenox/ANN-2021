import numpy as np                              
import matplotlib.pyplot as plt                    
from keras.utils import to_categorical            
from tensorflow.keras.models import Model          
from tensorflow.keras.datasets import cifar10        
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

pool_size = 2
kernel_size = 2
num_epochs = 20
batch_size = 100
drop_prob_1 = 0.2
drop_prob_2 = 0.5
conv_depth_1 = 32
conv_depth_2 = 64
hidden_size = 512
conv_depth_3 = 128
dropout = False

num_train, width, height, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.0
X_test /= 255.0

Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

inp = Input(shape=(width, height, depth))

conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', strides=(1, 1), activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
if (dropout):
    drop_1 = Dropout(drop_prob_1)(pool_1)
else:
    drop_1 = pool_1

conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', strides=(1, 1), activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', strides=(1, 1), activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
if (dropout):
    drop_2 = Dropout(drop_prob_1)(pool_2)
else:
    drop_2 = pool_2
conv_5 = Convolution2D(conv_depth_3, (kernel_size, kernel_size), padding='same', strides=(1, 1), activation='relu')(drop_2)
conv_6 = Convolution2D(conv_depth_3, (kernel_size, kernel_size), padding='same', strides=(1, 1), activation='relu')(conv_5)
pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_6)
if (dropout):
    drop_3 = Dropout(drop_prob_1)(pool_3)
else:
    drop_3 = pool_3
flat = Flatten()(drop_3)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_4 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_4)
model = Model(inp, out)

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=(X_test, Y_test))

result = model.evaluate(X_test, Y_test, verbose=0)
print('[Loss, Accuracy]: ', result)

x = range(1, num_epochs+1)
history_dict = hist.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'c', label='Train')
plt.plot(epochs, val_loss_values, 'g', label='Test')
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim(x[0], x[-1])
plt.legend()
plt.grid()
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'c', label='Train')
plt.plot(epochs, val_acc_values, 'g', label='Test')
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xlim(x[0], x[-1])
plt.legend()
plt.grid()
plt.show()