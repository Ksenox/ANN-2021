### Вариант 1

### Архитектура сети

layers.GRU(128, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
layers.LSTM(64, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
layers.Dense(1)

### Результат работы

val_loss: 0.007657153801992536

Графики представлены в файле plots.png