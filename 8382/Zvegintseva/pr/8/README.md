# Практика 8
## Вариант №6

Необходимо реализовать собственной CallBack, и провести обучение вашей модели из практического занятия №6 с написанным CallBack’ом. То, какой CallBack необходимо реализовать определяется вариантом.

Построение и сохранение таблицы со следующими данными: номер эпохи, номер наблюдения с наименьшей точностью классификации на заданной эпохе, к какому классу принадлежит наблюдение, точность классификации, значение ошибки.

Каждая строчка должна рассчитываться с заданным пользователем интервалом начиная с 0 эпохи, а также на самой последней

## Выполнение работы
В данной практике был реализован CallBack и проведено обучение модели из 6 практики(6 вариант).

Для записи данных был создан словарь, а также переданы нужные параметры для инициализации:
```
    def __init__(self, interval, dataTrain, labelTrain):
        super(CustomCallback, self).__init__()
        self.interval = interval
        self.dataTrain = dataTrain
        self.labelTrain = labelTrain
        self.table = {"epoch": [], "index": [], "class": [], "accuracy": [], "loss": []}
```
Был реализован метод on_epoch_end. Проверяется номер эпохи(соответствие величине интервала или последней эпохе) и в зависимости от этого классифицируются тренировочные данные. В связи с тем, что для оптимизации мы используем Softmax, то на выходе мы получаем три значения точности из которых нам необходимо выбрать нужное(минимальное при прохождении через функцию ).
Далее вычисляется к какому классу принадлежит наблюдение, точность классификации, значение ошибки.
В конце записываем нужные данные в словарь.
```
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0 or epoch == self.params["epochs"] - 1:

            self.table["epoch"].append(epoch)

            predictions = self.model.predict(self.dataTrain)
            ePred = np.zeros((len(predictions), 1))
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                ePred[i] = predictions[i][index]
            accCompare = 1 - abs(1 - ePred)
            minIndex = np.argmin(accCompare)
            self.table["index"].append(minIndex)

            if self.labelTrain[minIndex][0] == 1.0:
                self.table["class"].append("One")
            elif self.labelTrain[minIndex][1] == 1.0:
                self.table["class"].append("Two")
            elif self.labelTrain[minIndex][2] == 1.0:
                self.table["class"].append("Three")

            self.table["accuracy"].append(accCompare[minIndex])

            loss_ = keras.losses.get(self.model.loss)
            losses = np.asarray(loss_(self.labelTrain, predictions))
            self.table["loss"].append(losses[minIndex])
            df = pd.DataFrame(data=self.table)
            df.to_csv("CustomCallback.csv")
```