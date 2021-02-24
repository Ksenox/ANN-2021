import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from keras import layers
from keras.optimizers import Adam

test_fract = 0.2
val_fract = 0.2
target = 7
number_of_features = 7
encoding_dim = 500
number_of_neurons = 500
epochs = 100

def gen_data(size=500):
    X = norm.rvs(3,10,size)
    e = norm.rvs(0,0.3,size)
    data = np.ones((size,number_of_features))
    data[:,0] = X**2
    data[:, 1] = np.sin(X/2)
    data[:, 2] = np.cos(2 * X)
    data[:, 3] = X - 3
    data[:, 4] = -X
    data[:, 5] = np.abs(X)
    data[:, 6] = (X**3)/4
    data += e.reshape((e.shape[0],1))
    return data

def splitData(data):
    train_data = data[:int((1-test_fract)*data.shape[0])]
    val_data = train_data[int((1-val_fract)*train_data.shape[0]):train_data.shape[0]]
    train_data = train_data[:int((1-val_fract)*train_data.shape[0])]
    test_data = data[int((1-test_fract)*data.shape[0]):data.shape[0]]
    train_labels = np.ones((train_data.shape[0],1)) * target
    val_labels = np.ones((val_data.shape[0], 1)) * target
    test_labels = np.ones((test_data.shape[0], 1)) * target
    return train_data,val_data,test_data,train_labels,val_labels,test_labels

def saveCsv(data, name, columns=False):
    if columns:
        pd.DataFrame(data, columns=['labels', 'predict']).to_csv(name + ".csv")
    else:
        pd.DataFrame(data).to_csv(name + ".csv")

def submodel(inputs, name):
    hidden_layer = layers.Dense(number_of_neurons, activation='relu', name=name + "_1")(inputs)
    hidden_layer = layers.BatchNormalization(name=name + "_2")(hidden_layer)
    hidden_layer = layers.Dropout(0.5, name=name + "_3")(hidden_layer)
    return hidden_layer

def createModel(train_data, val_data, train_labels, val_labels,test_data):
    mean = np.mean(train_data,axis=0)
    std = np.std(train_data,axis=0)
    print(mean)

    std_train_data = (train_data - mean) / std
    std_val_data = (val_data - mean) / std
    std_test_data = (test_data - mean) / std

    input_layer = keras.Input(shape=(number_of_features,))
    hidden_layer_1 = submodel(input_layer,name="1")

    encoded = layers.Dense(encoding_dim, activation='relu', name="2_1")(hidden_layer_1)
    hidden_layer_2 = layers.BatchNormalization(name="2_2")(encoded)
    hidden_layer_2 = layers.Dropout(0.5, name="2_3")(hidden_layer_2)

    hidden_layer_3_1 = submodel(hidden_layer_2, name="3")

    hidden_layer_3_2 = submodel(hidden_layer_2, name="4")

    reg = layers.Dense(1,name="reg_out")(hidden_layer_3_1)
    decoded = layers.Dense(number_of_features,name="dec_out")(hidden_layer_3_2)

    model = keras.Model(inputs=input_layer, outputs=[reg, decoded])

    model.compile(loss={'reg_out': 'mean_squared_error',
                        'dec_out': 'mean_squared_error'},
                  optimizer=Adam(lr=0.001,decay=0.001/epochs))

    history = model.fit(std_train_data, {'reg_out': train_labels, 'dec_out': std_train_data},
                        epochs=epochs,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(std_val_data, {'reg_out':val_labels, 'dec_out':std_val_data}))

    encoder = keras.Model(input_layer, encoded)

    regress_input = keras.Input(shape=(encoding_dim,))
    hidden_layer = layers.BatchNormalization()(regress_input)
    hidden_layer = layers.Dropout(0.5)(hidden_layer)
    hidden_layer = layers.Dense(number_of_neurons, activation='relu')(hidden_layer)
    hidden_layer = layers.BatchNormalization()(hidden_layer)
    hidden_layer = layers.Dropout(0.5)(hidden_layer)
    regress_output = layers.Dense(1)(hidden_layer)
    regression = keras.Model(regress_input, regress_output)


    regression.layers[1].set_weights(model.get_layer("2_2").get_weights())
    regression.layers[2].set_weights(model.get_layer("2_3").get_weights())
    regression.layers[3].set_weights(model.get_layer("3_1").get_weights())
    regression.layers[4].set_weights(model.get_layer("3_2").get_weights())
    regression.layers[5].set_weights(model.get_layer("3_3").get_weights())
    regression.layers[6].set_weights(model.get_layer("reg_out").get_weights())

    decoder_input = keras.Input(shape=(encoding_dim,))
    hidden_layer = layers.BatchNormalization()(decoder_input)
    hidden_layer = layers.Dropout(0.5)(hidden_layer)
    hidden_layer = layers.Dense(number_of_neurons, activation='relu')(hidden_layer)
    hidden_layer = layers.BatchNormalization()(hidden_layer)
    hidden_layer = layers.Dropout(0.5)(hidden_layer)
    decoder_output = layers.Dense(number_of_features)(hidden_layer)
    decoder = keras.Model(decoder_input, decoder_output)

    decoder.layers[1].set_weights(model.get_layer("2_2").get_weights())
    decoder.layers[2].set_weights(model.get_layer("2_3").get_weights())
    decoder.layers[3].set_weights(model.get_layer("4_1").get_weights())
    decoder.layers[4].set_weights(model.get_layer("4_2").get_weights())
    decoder.layers[5].set_weights(model.get_layer("4_3").get_weights())
    decoder.layers[6].set_weights(model.get_layer("dec_out").get_weights())

    encoder.save("encoder.h5")
    regression.save("regression.h5")
    decoder.save("decoder.h5")

    encods = encodeData(std_test_data, encoder)
    decodeData(encods, decoder, mean, std)
    computeRegression(encods, regression)

def encodeData(test_data, encoder):
    preds = encoder.predict(test_data)
    saveCsv(preds, "encoded")
    return preds

def decodeData(encods, decoder, mean, std):
    preds = decoder.predict(encods) * std + mean
    saveCsv(preds, "decoded")

def computeRegression(encods, regression):
    preds = regression.predict(encods)
    labels_preds = np.ones((encods.shape[0],2))
    labels_preds[:,0] = target
    labels_preds[:,1] *= preds.reshape((preds.shape[0],))
    saveCsv(labels_preds, "regression", columns=True)

data = gen_data()
saveCsv(data, "dataset")
train_data, val_data, test_data, train_labels, val_labels, test_labels = splitData(data)
createModel(train_data, val_data, train_labels, val_labels, test_data)