# Задание (Вариант 4)

Необходимо построить сверточную нейронную сеть, которая будет классифицировать черно-белые изображения с простыми геометрическими фигурами на них.
К каждому варианту прилагается код, который генерирует изображения.
Для генерации данных необходимо вызвать функцию gen_data, которая возвращает два тензора:

1. Тензор с изображениями ранга 3
2. Тензор с метками классов

Обратите внимание:

* Выборки не перемешаны, то есть наблюдения классов идут по порядку
* Классы характеризуются строковой меткой
* Выборка изначально не разбита на обучающую, контрольную и тестовую
* Скачивать необходимо оба файла. Подключать файл, который начинается с var (в нем и находится функция gen_data)

Классификация изображений с крестом или с линией (может быть горизонтальной или вертикальной)

## Выполнение работы

Были сгенерированы данные с помощью соответствующей функции. 
С помощью `LabelEncoder` были закодированы метки.
Функция `train_test_split` позволяет разбить данные на тренировочные и тестовые, в случайном порядке.

Модель выглядит следующим образом:

![](https://i.ibb.co/GQ6Vn7Z/model.png)

* Размеры ядра в сверточных слоях -- 7x7.

* Размеры подвыборки в слоях подвыборки -- 5x5.

* Вероятность исключения нейрона у обоих отсеивающих слоев -- 40%.

* 64 нейрона в полносвязном слое.

    
## Тестирование

При обучении за 10 эпох были получены следующие результаты:

|Train|Validation|Test|
|---|---|---|
|0.9886|0.9412|0.9879|

![](https://i.ibb.co/7K2sPh9/Accuracy.png)

![](https://i.ibb.co/fvNWQ2C/Loss.png)

Модель с неплохой точностью (98.8%) различает линии и кресты.