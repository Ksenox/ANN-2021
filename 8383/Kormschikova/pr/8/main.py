
import matplotlib.pyplot as plt  # импорт модуля для графиков
from keras.models import Model
from keras.callbacks import Callback
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import var3
from time import gmtime, strftime

epochs_to_save = []
prefix = []
num_epochs = 25

class ModelSaver(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epochs_to_save.count(epoch+1) > 0:
            modelName = strftime("%d-%m-%y_", gmtime())+prefix[0]+"_"+str(epoch+1)+".h5"
            self.model.save(modelName)
            print(modelName+" saved!")

def interface():
    print("Enter the epoch numbers separated by a space ")
    n_epoch = input()
    tmp = n_epoch.split(" ")
    for i in tmp:
        try:
            num = int(i)
            if num_epochs >= num > 0 and epochs_to_save.count(num) == 0:
                epochs_to_save.append(num)
        except ValueError:
            continue
    epochs_to_save.sort()
    print("Epochs to save:")
    print(epochs_to_save)
    print("Enter prefix to filename")
    prefix.append(input())


interface()
train_size = 1000
test_size = 400
data_size = train_size + test_size
img_size = 50
data, labels = var3.gen_data(data_size, img_size)
data, labels = shuffle(data, labels)  # перемешивание

# переход от текстовых меток к категориальному вектору
encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
encoded_labels = to_categorical(encoded_labels)

train_data = data[:train_size]
train_labels = encoded_labels[:train_size]
val_split = 0.1

test_data = data[train_size:]
test_labels = encoded_labels[train_size:]
batch_size = 50
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 64




inp = Input(shape=(img_size, img_size, 1))
# Два сверточных слоя, слой пуллинга, слой dropout
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

flat = Flatten()(drop_1)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(2, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks_list = [ModelSaver()]

history = model.fit(train_data, train_labels, callbacks=callbacks_list, batch_size=batch_size, epochs=num_epochs,
                    verbose=1, validation_split=val_split)

history_dict = history.history
model.evaluate(test_data, test_labels, verbose=1)
# график ошибки
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'b-.', label='Training loss')
plt.plot(epochs, val_loss_values, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# график точности
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'b-.', label='Training acc')
plt.plot(epochs, val_acc_values, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
