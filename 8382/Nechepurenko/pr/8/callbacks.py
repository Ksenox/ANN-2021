from typing import Iterable

import numpy as np
from PIL import Image
from tensorflow.python.keras.callbacks import Callback


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
