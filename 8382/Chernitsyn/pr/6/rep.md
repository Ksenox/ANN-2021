# PR6 Черницын П.А. Вар.1

### Генерация

С помощью функций, данных в задании, генерируются изображения квадрата и круга.
Заем они перемешиваются. 

```py
    label_c1 = np.full([c1, 1], 'Square')
    data_c1 = np.array([gens.gen_rect(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Circle')
    data_c2 = np.array([gens.gen_circle(img_size) for i in range(c2)])
])
```

### Модель


```py
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
])
```

Параметры обучения:

```py
EPOCHS = 10
BATCH_SIZE = 32
history = model.fit(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(valid_data, valid_labels))
```

### Result

Графики: graphic1.png и graphic2.png

Epoch 10/10
120/120 [==============================] - 14s 116ms/step - loss: 0.0098 - accuracy: 0.9977 - val_loss: 0.0021 - val_accuracy: 0.9990
38/38 [==============================] - 1s 25ms/step - loss: 0.0062 - accuracy: 0.9967