from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np 
import matplotlib.pyplot as plt
import var4


# Генерация датасета из последовательности
def gen_data_from_sequence(seq_len = 1006, lookback = 10):
    seq = var4.gen_sequence(seq_len)  # генерируем последовательность нужной длины
    past = np.array([[[seq[j]] for j in range(i,i+lookback)] for i in range(len(seq) - lookback)]) # составляем массив из векторов длины lookback, содержащих значения последовательности в интервале [i, i+lookback)
    future = np.array([[seq[i]] for i in range(lookback,len(seq))])  # значение последовательности в момент i+lookback
    return (past, future)


# Рисование графика потерь
def draw_loss_plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(len(loss)),loss,label="train_loss",c="blue")
    plt.plot(range(len(val_loss)),val_loss,label="val_loss", c="red")
    plt.legend()
    plt.title("Потери на тренировочных и валидационных данных")
    plt.show()


# Рисование предсказанной последовательности
def draw_predicted_results(predicted_res):
    pred_length = range(len(predicted_res))
    plt.plot(pred_length,predicted_res,label="Предсказанное",c="red")
    plt.plot(pred_length,test_res,label="Ожидаемое",c="blue")
    plt.title("Предсказанные и ожидаемые результаты")
    plt.legend()
    plt.show()


# Подготовка данных
data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 9

train_data, train_res = data[:train_size], res[:train_size]
test_data, test_res = data[train_size:], res[train_size:]

# Создание модели
model = Sequential()
model.add(layers.GRU(32,recurrent_activation='sigmoid',input_shape=(None,1),return_sequences=True))
model.add(layers.GRU(32,activation='relu',input_shape=(None,1),return_sequences=True))
model.add(layers.GRU(32,input_shape=(None,1),recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')
history = model.fit(train_data, train_res, epochs=50, validation_split=0.15)

draw_loss_plot(history)

# Предсказание результатов
predicted_results = model.predict(test_data)
draw_predicted_results(predicted_results)

