# Практическая работа №8

## Задание.
Необходимо реализовать собственной CallBack, и провести обучение вашей модели из практического занятия №6 с написанным CallBack’ом. То, какой CallBack необходимо реализовать определяется вариантом.

Вариант 2:
Построение и сохранение карты признаков на заданных пользователем эпохах. Карта признаков - ядро свертки представленное в виде изображения. Название карты признака должно иметь вид <номер слоя>_<номер ядра в слое>_<номер эпохи>

## Выполнение работы.
Унаследуемся от класса Callback и переопределим
метод `on_epoch_end`, чтобы обработчик вызывался после каждой эпохи.

```python
class FeatureMapForEpochCallback(Callback):
    def __init__(self, epochs_list: Iterable[int]):
        super().__init__()
        self._epochs_list = epochs_list

    def on_epoch_end(self, epoch, logs=None):
        if epoch not in self._epochs_list:
            return
        for layer_index, layer in self._get_conv_layers():
            weights = layer.weights[0]
            kernel_count = weights.shape[3]
            for i in range(weights.shape[2]):
                for kernel_index in range(kernel_count):
                    Image \
                        .fromarray(np.uint8(255 * weights[:, :, i, kernel_index])) \
                        .save(f'feature-maps/{layer_index + 1}_{i * kernel_count + kernel_index + 1}_{epoch}.png')

    def _get_conv_layers(self) -> Iterable:
        for layer_index, layer in enumerate(self.model.layers):
            if layer.name.startswith("conv2d"):
                yield layer_index, layer
```

В конструкторе сохраним список интересующих эпох.
В обработчике проверяем, является ли текущая эпоха одной
из необходимых к обработке.

Проитерируемся по `Convolution2D` слоям и сохраним представления
его ядер в виде изображения.

Для использования коллбэка, передадим его в функцию `fit` и
укажем 1 эпоху.
```python
history = model.fit(train_x, train_y, batch_size=20, epochs=10, validation_split=0.1, callbacks=[FeatureMapForEpochCallback([1])])
```

На выходе получается достаточно много изображений, у последнего
было название `3_2048_1.png`, но дабы не засорять github, большинство изображений
были удалены.