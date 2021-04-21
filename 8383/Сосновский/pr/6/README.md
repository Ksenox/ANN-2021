# Практическое задание №6, Вариант 6
Необходимо построить сверточную нейронную сеть, которая будет классифицировать черно-белые изображения с простыми геометрическими фигурами на них.

К каждому варианту прилагается код, который генерирует изображения.

Для генерации данных необходимо вызвать функцию gen_data, которая возвращает два тензора:

1) Тензор с изображениями ранга 3
2) Тензор с метками классов

Обратите внимание:

1) Выборки не перемешаны, то есть наблюдения классов идут по порядку
2) Классы характеризуются строковой меткой
3) Выборка изначально не разбита на обучающую, контрольную и тестовую
4) Скачивать необходимо оба файла. Подключать файл, который начинается с var (в нем и находится функция gen_data)
 
# Описание решения
Была построена архитектура сверточной нейронной сети. Она состоит из 8 слоёв:
```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=(height, width, depth)))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(rate=drop_prob_1))
model.add(Flatten())
model.add(Dense(dense_size_1, activation='relu'))
model.add(Dropout(rate=drop_prob_2))
model.add(Dense(dense_size_2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
```
В качестве функции потерь была выбрана функция бинарной кроссэнтропии

В качестве оптимизатора был взят оптимизатор Adam

Были взяты следующие численные параметры обучения:
```python 
batch_size = 16
num_epochs = 20
```
Полученные оценочные значения нейросети:
```
loss: 0.1459
accuracy: 0.9720
```

Процесс обучения по всем эпохам: 
```python
Epoch 1/20
141/141 [==============================] - 8s 54ms/step - loss: 0.8087 - accuracy: 0.5970 - val_loss: 0.2170 - val_accuracy: 0.9253
Epoch 2/20
141/141 [==============================] - 7s 50ms/step - loss: 0.2373 - accuracy: 0.9099 - val_loss: 0.1623 - val_accuracy: 0.9560
Epoch 3/20
141/141 [==============================] - 7s 51ms/step - loss: 0.1557 - accuracy: 0.9399 - val_loss: 0.1811 - val_accuracy: 0.9387
Epoch 4/20
141/141 [==============================] - 6s 46ms/step - loss: 0.1724 - accuracy: 0.9252 - val_loss: 0.1637 - val_accuracy: 0.9480
Epoch 5/20
141/141 [==============================] - 8s 60ms/step - loss: 0.1309 - accuracy: 0.9538 - val_loss: 0.1267 - val_accuracy: 0.9440
Epoch 6/20
141/141 [==============================] - 8s 56ms/step - loss: 0.1494 - accuracy: 0.9406 - val_loss: 0.1300 - val_accuracy: 0.9573
Epoch 7/20
141/141 [==============================] - 9s 61ms/step - loss: 0.1034 - accuracy: 0.9629 - val_loss: 0.1160 - val_accuracy: 0.9573
Epoch 8/20
141/141 [==============================] - 6s 46ms/step - loss: 0.0977 - accuracy: 0.9570 - val_loss: 0.1152 - val_accuracy: 0.9560
Epoch 9/20
141/141 [==============================] - 6s 45ms/step - loss: 0.0779 - accuracy: 0.9728 - val_loss: 0.1405 - val_accuracy: 0.9333
Epoch 10/20
141/141 [==============================] - 8s 56ms/step - loss: 0.0988 - accuracy: 0.9591 - val_loss: 0.1866 - val_accuracy: 0.9200
Epoch 11/20
141/141 [==============================] - 7s 47ms/step - loss: 0.0946 - accuracy: 0.9627 - val_loss: 0.1167 - val_accuracy: 0.9520
Epoch 12/20
141/141 [==============================] - 7s 47ms/step - loss: 0.0550 - accuracy: 0.9829 - val_loss: 0.2649 - val_accuracy: 0.9107
Epoch 13/20
141/141 [==============================] - 7s 52ms/step - loss: 0.0639 - accuracy: 0.9760 - val_loss: 0.1361 - val_accuracy: 0.9640
Epoch 14/20
141/141 [==============================] - 7s 49ms/step - loss: 0.0385 - accuracy: 0.9904 - val_loss: 0.1004 - val_accuracy: 0.9653
Epoch 15/20
141/141 [==============================] - 8s 57ms/step - loss: 0.0532 - accuracy: 0.9848 - val_loss: 0.1005 - val_accuracy: 0.9613
Epoch 16/20
141/141 [==============================] - 8s 58ms/step - loss: 0.0496 - accuracy: 0.9820 - val_loss: 0.1668 - val_accuracy: 0.9507
Epoch 17/20
141/141 [==============================] - 8s 54ms/step - loss: 0.0502 - accuracy: 0.9786 - val_loss: 0.1225 - val_accuracy: 0.9427
Epoch 18/20
141/141 [==============================] - 7s 50ms/step - loss: 0.0335 - accuracy: 0.9867 - val_loss: 0.1056 - val_accuracy: 0.9587
Epoch 19/20
141/141 [==============================] - 7s 47ms/step - loss: 0.0289 - accuracy: 0.9885 - val_loss: 0.1046 - val_accuracy: 0.9733
Epoch 20/20
141/141 [==============================] - 7s 50ms/step - loss: 0.0218 - accuracy: 0.9938 - val_loss: 0.1000 - val_accuracy: 0.9680
32/32 [==============================] - 0s 12ms/step - loss: 0.1459 - accuracy: 0.9720
```

Полученные графики ошибок и точности:
![Image alt](https://i.ibb.co/NYsKb3s/image.png)

![Image alt](https://i.ibb.co/2sRG92z/image.png)

