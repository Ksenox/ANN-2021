# Практическое задание 6
Ларин Антон  
Гр. 8383
  
## Условие задачи

> Вариант 5   

Необходимо построить сверточную нейронную сеть, которая будет классифицировать черно-белые изображения с простыми геометрическими фигурами на них.  
  
К каждому варианту прилагается код, который генерирует изображения.

Для генерации данных необходимо вызвать функцию gen_data, которая возвращает два тензора:
1. Тензор с изображениями ранга 3
2. Тензор с метками классов   

Обратите внимание:
- Выборки не перемешаны, то есть наблюдения классов идут по порядку
- Классы характеризуются строковой меткой
- Выборка изначально не разбита на обучающую, контрольную и тестовую
- Скачивать необходимо оба файла. Подключать файл, который начинается с var (в нем и находится функция gen_data)

#### Вариант 5
Классификация изображений с прямоугольником или не закрашенным кругом

## Выполнение работы

Модель состоит из двух блоков, каждый по два слоя свортки, слой MaxPooling и слой Dropout
Карта признаков сплющивается(Flatted) и подается на вход полносвязному слою, далее в Dropout и в выходной слой


Модель имеет следующий вид:
```python
inp = Input(shape=(img_size, img_size,1))

# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
layer = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
layer = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(layer)
layer = MaxPooling2D(pool_size=(pool_size, pool_size))(layer)
layer = Dropout(drop_prob_1)(layer)

# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
layer = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
layer = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(layer)
layer = MaxPooling2D(pool_size=(pool_size, pool_size))(layer)
layer = Dropout(drop_prob_1)(layer)


flat = Flatten()(layer)

layer = Dense(hidden_size, activation='relu')(flat)
layer = Dropout(drop_prob_2)(layer)
out = Dense(num_classes, activation='softmax')(layer)
```

Далее модель компилируется и обучается на входных данных

```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

H = model.fit(data,labels,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=1,
              validation_split=.1)
```
Модель обучена и провалидирована на 400 экземплярах входных данных и 200 эпохах, с batch size = 32 

## Полученные результаты

```
...
Epoch 200/200
57/57 [==============================] - 1s 11ms/step - loss: 3.6944e-09 - accuracy: 1.0000 - val_loss: 1.7881e-09 - val_accuracy: 1.0000
63/63 [==============================] - 0s 4ms/step - loss: 5.3644e-10 - accuracy: 1.0000

[5.364417243747255e-10, 1.0]
```  
Результат удовлетворительный