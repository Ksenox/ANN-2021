from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
from functions import *

# input data
studData = np.genfromtxt('stud.csv', delimiter=',')
validData = np.genfromtxt('valid.csv', delimiter=',')

# parse
studX = studData[:, 1:]
studVal = studData[:, :1]
validX = validData[:, 1:]
validVal = validData[:, :1]

# encode
encodedStudX = encode(studX)
encodedValidX = encode(validX)

# Layers
# input
firstInput = Input(shape=(1,), name='firstInput')
# encoding
encodingOutput = Dense(1, name="eo")(firstInput)
# decoding
decodingOutput = Dense(1, name="do")(encodingOutput)
# regression
regressionLayer = Dense(32, activation='relu')(encodingOutput)
regressionOutput = Dense(1, name='regressionOutput')(regressionLayer)


# main model
model = Model(inputs=[firstInput], outputs=[
              regressionOutput, encodingOutput, decodingOutput])
model.compile(optimizer='RMSprop', loss='MeanSquaredError', metrics='MeanAbsoluteError')
model.fit([studX], [studVal, encodedStudX, studX],
          epochs=150, batch_size=10, validation_split=0, verbose=2)
# Models
# encoding
encodingModel, encodingPrediction = modelCreate(
    firstInput, encodingOutput, validX)
encodingModel.save('models/encoding.h5')
np.savetxt('results/encoding.csv', np.hstack(
    (encodedValidX, encodingPrediction)), delimiter=',')
# decoding
decodingModel, decodingPrediction = modelCreate(
    firstInput, decodingOutput, validX)
decodingModel.save('models/decoding.h5')
np.savetxt('results/decoding.csv', np.hstack(
    (validX, decodingPrediction)), delimiter=',')
# regression
regressionModel, regressionPrediction = modelCreate(
    firstInput, regressionOutput, validX)
regressionModel.save('models/regression.h5')
np.savetxt('results/regression.csv', np.hstack(
    (validVal, regressionPrediction)), delimiter=',')
