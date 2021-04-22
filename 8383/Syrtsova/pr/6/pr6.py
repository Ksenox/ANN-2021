import var1
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_data():
    data, labels = var1.gen_data(1000, 50)
    data = data.reshape(data.shape[0], 50, 50, 1)
    encoder = LabelEncoder()
    encoder.fit(labels.ravel())
    labels = encoder.transform(labels.ravel())
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    return train_data, test_data, train_labels, test_labels


def build_model():
    model = Sequential()
    model.add(Input(shape=(50, 50, 1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = build_model()
train_data, test_data, train_labels, test_labels = get_data()
history = model.fit(train_data, train_labels, batch_size=20, epochs=10, validation_data=(test_data, test_labels))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.evaluate(test_data, test_labels)
