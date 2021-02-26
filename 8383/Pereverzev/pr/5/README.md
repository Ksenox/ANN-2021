# 8383 Pereverzev Dmitriy pr5 v2.3

Необходимо в зависимости от варианта сгенерировать датасет и сохранить его в формате csv.

Построить модель, которая будет содержать в себе автокодировщик и регрессионную модель.
Обучить модель и разбить обученную модель на 3: Модель кодирования данных (Входные данные -> Закодированные данные), модель декодирования данных (Закодированные данные -> Декодированные данные), и регрессионную модель (Входные данные -> Результат регрессии).

В качестве результата представить исходный код, сгенерированные данные в формате csv, кодированные и декодированные данные в формате csv, результат регрессии в формате csv (что должно быть и что выдает модель), и сами 3 модели в формате h5.

### Вариант 2.3

X: `[-5:10]` 

e: `[0:0.3]` 
|Признак|...|     3     |...|
|-------|---|-----------|---|
|Формула|...| sin(3x)+e |...|

### Модель сети

```py
# main model
model = Model(inputs=[firstInput], outputs=[
              regressionOutput, encodingOutput, decodingOutput])
model.compile(optimizer='RMSprop', loss='MeanSquaredError', metrics='MeanAbsoluteError')
model.fit([studX], [studVal, encodedStudX, studX],
          epochs=150, batch_size=10, validation_split=0, verbose=2)
```
#### Слои

```py
# Layers
# input
firstInput = Input(shape=(1,), name='firstInput')
# encoding
encodingOutput = Dense(1, name="eo")(firstInput)
# decoding
decodingOutput = Dense(1, name="do")(encodingOutput)
# regression
regressionLayer = Dense(32, activation='relu')(encodingOutput)
regressionOutput = Dense(1, name='regressionOutput')(regressionLayer)

```
### Генерация данных
```py
def v2_3(X):
    return[(3*x[0]) + np.random.rand(1) * e[1] for x in X]
def gen(size):
    localX = [(np.random.rand(1) * (x[1]-x[0]))+x[0]
              for idx in range(size)]
    localVal = v2_3(localX)
    return np.hstack((np.asarray(localX), np.asarray(localVal)))
```