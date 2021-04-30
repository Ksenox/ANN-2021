# Задание

Необходимо построить сверточную нейронную сеть, которая будет классифицировать черно-белые изображения с простыми геометрическими фигурами на них.
К каждому варианту прилагается код, который генерирует изображения.
Для генерации данных необходимо вызвать функцию gen_data, которая возвращает два тензора:

1. Тензор с изображениями ранга 3
2. Тензор с метками классов

# Вариант 4
Классификация изображений с крестом или с линией (может быть горизонтальной или вертикальной)

# Выполнение

Вначале используя функцию gen_data были сгенерированы данные, затем были закодированы метки, а данные разделены на тестовые и треинировочные случайным образом.

Затем была построена модель нейронной сети, состоящая из двух сверточных слоев Convolution2D, слоя субдискретизации MaxPooling2D, слоя изменяющего форму данных Flatten и двух полносвязных слоев Dense.
В сверточных слоях используется ядро свертки размера 7 на 7 и применяют к изображению 32 фильтра, субдискретизация по окнам размера 5 на 5.
На выходном слое функция активации sigmoid позволяет выдать вероятность принадлежности изображения к одному из классов. 

Затем модель обучается в течение 8 эпох пакетами по 10 образцов, 25% входных данных отданы под валидацию.
Модель очень быстро достигает высокой точности на тренировочных и валидационных данных.
Посмотрим результаты оценки модели:

4/4 [==============================] - 0s 31ms/step - loss: 4.9223e-04 - accuracy: 1.0000

При 5 запусках программы всего один раз сеть выдала итоговую точность 98%, а не 100%.
Результат обучения можно считать очень успешным.
