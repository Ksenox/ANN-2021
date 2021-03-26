# Задание
**Задача 2**

Написать функцию преобразовывающую вектор чисел в матрицу бинарных представлений.

Примечания:
+ Необходимо использовать модуль numpy
+ Все данные должны считываться из файла в виде массива numpy
+ Результаты необходимо сохранять в файл

# Выполнение работы

Исходные данные задаются задаются в файле `input.csv` и представляют собой набор целых чисел, разделенных знаком `;`. 
Данные считываются из файла в виде одномерного массива при помощи метода `numpy.fromfile`. 
Преобразование в матрицу бинарных представлений осуществляется методом `numpy.unpackbits`, 
который возвращает бинарное представление элементов массива вдоль заданной оси в виде вектора. 
Метод `numpy.unpackbits`, однако, работает только с числами в формате `uint8`, поэтому, 
чтобы функция могла работать с большими числами, необходимо предварительно изменить представление считанных из файла чисел.

Сначала исходный вектор транспонируется. 
Затем исходные числа представляются в формате `uint8` при помощи метода `numpy.ndarray.view`. 
Таким образом получаем матрицу, каждая строка которой содержит представление числа в формате `uint8`. 
При помощи метода `numpy.flip` порядок элементов в строках матрицы меняется на противоположный, 
так  метод `numpy.ndarray.view` возвращает байты числа в обратном порядке.

Из полученной матрицы уже можно получить бинарное представление чисел с помощью метода `numpy.unpackbits`, 
однако, так как при считывании исходного вектора чисел из файла явно не указывается размер этих чисел, 
в полученной матрице может оказаться много незначащих нулей (или единиц), что затрудняет чтение полученного результата. 
Чтобы этого избежать, была реализована вспомогательная функция `trim_matrix`, которая "обрезает" матрицу представлений 
чисел в формате `uint8` слева по наибольшему по модулю числу.

Искомый результат, возвращаемый методом `numpy.unpackbits`, сохраняется в файл `output.csv`, а также возвразается реализованной функцией.


# Тестирование
**input.csv**

66000;0;300;-300;-12;12;3;6;14;7;128

**output.csv**

0;0;0;0;0;0;0;1;0;0;0;0;0;0;0;1;1;1;0;1;0;0;0;0

0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0

0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;0;0;1;0;1;1;0;0

1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;1;1;0;1;0;1;0;0

1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;1;0;0

0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;1;0;0

0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;1

0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;1;0

0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;1;1;0

0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;1;1

0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1;0;0;0;0;0;0;0