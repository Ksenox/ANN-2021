from keras.models import Model
from keras.layers import Dense
import numpy as np
from math import asin


def encode(a):
    return a*-1


def layersTemp(inp):
    lname = str(np.random.rand(1)[0])
    layer = Dense(32, activation='relu')(inp)
    layer = Dense(16, activation='relu',
                  name=f"layer1{lname}")(layer)
    layer = Dense(16, activation='relu',
                  name=f"layer2{lname}")(layer)
    outp = Dense(1, activation='relu',
                 name=f"layer3{lname}")(layer)
    return layer, outp


def modelCreate(inp, outp, val):
    tmodel = Model(inputs=[inp], outputs=[outp])
    tprediction = tmodel.predict(val)
    return tmodel, tprediction
