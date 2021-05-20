from keras.callbacks import Callback
from keras.layers import Convolution2D
import numpy as np
from PIL import Image
import os


class FeatureMapSaver(Callback):
    def __init__(self, epochs: list):
        super().__init__()
        self.epochs = epochs
        os.makedirs('./feature_maps', exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch not in self.epochs:
            return

        for index, layer in enumerate(self.model.layers):
            if type(layer) == Convolution2D:
                weights = layer.weights[0]

                for i in range(weights.shape[2]):
                    for j in range(weights.shape[3]):
                        image = Image.fromarray(np.uint8(weights[:, :, i, j] * 255))
                        image.save(f'./feature_maps/{index}_{i * weights.shape[3] + j}_{epoch}.png')
