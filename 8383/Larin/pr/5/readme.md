# Практическое задание 5
Ларин Антон  
Гр. 8383

## Условие задачи

> Вариант 4.4

Необходимо в зависимости от варианта сгенерировать датасет и сохранить его в формате csv.
Построить модель, которая будет содержать в себе автокодировщик и регрессионную модель.
Обучить модель и разбить обученную модель на 3: Модель кодирования данных (Входные данные -> Закодированные данные), модель декодирования данных (Закодированные данные -> Декодированные данные), и регрессионную модель (Входные данные -> Результат регрессии).  

В качестве результата представить исходный код, сгенерированные данные в формате csv, кодированные и декодированные данные в формате csv, результат регрессии в формате csv (что должно быть и что выдает модель), и сами 3 модели в формате h5.  

#### Вариант 4.4

## Выполнение работы

Модель имеет следующий вид:
```python
def get_encoder(input):
    hiden_encode_layer = Dense(n_of_neurons, activation='relu')(input)
    hiden_encode_layer = Dense(int(n_of_neurons/2), activation='relu')(hiden_encode_layer)
    encoder_output =  Dense(int(encode_dim), activation='relu', name='encoder_output')(hiden_encode_layer)
    return encoder_output

def get_decoder(input):
    hiden_decode_layer = Dense(n_of_neurons, activation='relu', name="decoder_1")(input)
    hiden_decode_layer = Dense(n_of_neurons, activation='relu', name="decoder_2")(hiden_decode_layer)
    decoder_output = Dense(encode_dim, activation='relu', name='decoder_output')(hiden_decode_layer)
    return decoder_output

def get_reg(input):
    hiden_reg_layer = Dense(n_of_neurons, activation='relu')(input)
    hiden_reg_layer = Dense(n_of_neurons, activation='relu')(hiden_reg_layer)
    reg_output = Dense(1, name='reg_output')(hiden_reg_layer)
    return reg_output

#...

main_input = Input(shape=(6,), name='main_input')
encoder = get_encoder(main_input)
decoder = get_decoder(encoder)
reg = get_reg(encoder)

full_model = Model(inputs=main_input, outputs=[reg, decoder])
```
Модель имеет один вход и два выхода  

При выполнении производится генерация датасета и сохранение его в .csv файлы
  
Далее датасет загружается из файлов и на нем обучается вся модель

```python
full_model.fit(train_data, [train_labels, train_data], epochs=epochs, batch_size=16, validation_split=0.1, verbose=0)
```

После того как вая можель обучена она разбивается на три модели: энкодер, декодер и можель регрессии

```python
encoder_model = Model(inputs=main_input, outputs=encoder)
reg_model = Model(inputs=main_input, outputs=reg)
decoder_model = Model(inputs=main_input, outputs=decoder)
```


Данные модели созраняются в файл.
Затем модели загружаются из файла и по ним прогоняются тестовые данные. Результат сохраняется в .csv файлы

## Полученные результаты
Оценка модели показала сделующие результаты:
```
loss: 24.5082 - reg_output_loss: 0.0140 - decoder_output_loss: 24.4941 - reg_output_mae: 0.0751 - decoder_output_mae: 2.4024
```

Точность регрессии очень хорошая. Декодирование дает точность хуже, однако достаточно хорошую