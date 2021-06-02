import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pickle

#read and analyze
df = pd.read_csv('sample_data/Concrete Compressive Strength.csv')
# df.head()
# df.describe()
# df.info()

#split to x and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#split to train and test; to normal
for column in X.columns:
    X[column] += 1
    X[column] = np.log(X[column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(X_train.head())

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train[:5, :])

#configure callbacks
# %load_ext tensorboard
# !rm -rf ./logs/
filepath = "-{epoch:02d}-{mae:.4f}.hdf5"
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)


#first model
relu2l = Sequential()
relu2l.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
relu2l.add(Dense(64, activation='relu'))
relu2l.add(Dense(1))
relu2l.compile(optimizer='adam', loss='mse', metrics=['mae'])
h1 = relu2l.fit(
    X_train, y_train, epochs=100, batch_size=1,
    validation_split=0.2, verbose=1, callbacks=[
        tensorboard_callback,
        ModelCheckpoint("relu2l" + filepath, monitor='mae', verbose=1, save_best_only=True, mode='min'),
    ]
)
print(relu2l.evaluate(X_test, y_test))

#second model
relu3l = Sequential()
relu3l.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer='glorot_normal'))
relu3l.add(Dense(96, activation='relu', kernel_initializer='glorot_normal'))
relu3l.add(Dense(64, activation='relu', kernel_initializer='glorot_normal'))
relu3l.add(Dense(1))
relu3l.compile(optimizer='adam', loss='mse', metrics=['mae'])
h2 = relu3l.fit(
    X_train, y_train, epochs=100, batch_size=1,
    validation_split=0.2, verbose=1, callbacks=[
        tensorboard_callback,
        ModelCheckpoint("relu3l" + filepath, monitor='mae', verbose=1, save_best_only=True, mode='min')
    ]
)
print(relu3l.evaluate(X_test, y_test))

# non-ann model
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
print(mean_absolute_error(y_test, random_forest.predict(X_test)))
# with open('rf.pickle', 'wb') as f:
#     pickle.dump(random_forest, f)


def get_ensembled():
    return 0.5 * random_forest.predict(X_test) + 0.2 * relu2l.predict(X_test)[:, 0] + 0.3 * relu3l.predict(X_test)[:, 0]


# relu2l.load_weights("relu2l-93-2.2260.hdf5")
# relu3l.load_weights("relu3l-99-1.8773.hdf5")
# with open('rf.pickle', 'rb') as f:
#     random_forest = pickle.load(f)

#evaluation
print(relu2l.evaluate(X_test, y_test))
print(relu3l.evaluate(X_test, y_test))
print(mean_absolute_error(y_test, random_forest.predict(X_test)))
print(mean_absolute_error(y_test, get_ensembled()))

# %tensorboard --logdir logs/
