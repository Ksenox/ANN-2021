Вариант 5

Модель
```python
model.add(layers.GRU(128, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(layers.LSTM(64, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
model.add(layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
model.add(layers.Dense(1))
```

Результат в `myplot.png`