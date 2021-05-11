from tensorflow.keras.models import Sequential
from keras import layers
from var4 import *

# data, res, seq = gen_data_from_sequence()

data, res, seq = gen_data_from_sequence()

# draw_sequence(seq)

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size+val_size], res[train_size:train_size+val_size]
test_data, test_res = data[train_size+val_size:], res[train_size+val_size:]

model = Sequential()
model.add(layers.GRU(40,recurrent_activation='sigmoid',input_shape=(None,1),return_sequences=True))
model.add(layers.LSTM(40,activation='relu',input_shape=(None,1),return_sequences=True,dropout=0.2))
model.add(layers.GRU(32,input_shape=(None,1),recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse')
history = model.fit(train_data,train_res,epochs=50,validation_data=(val_data, val_res))

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)),loss, label="loss")
plt.plot(range(len(val_loss)),val_loss, label="val_loss")
plt.xlabel("epochs")
plt.xlabel("loss")
plt.legend()
plt.show()

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length,predicted_res, label= "predicted_res")
plt.plot(pred_length,test_res, label="test_res")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()