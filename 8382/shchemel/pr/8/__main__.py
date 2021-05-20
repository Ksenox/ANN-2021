from datetime import datetime

import numpy as np
from keras.layers import (Convolution2D, Dense, Dropout, Flatten, Input,
                          MaxPooling2D)
from keras.models import Model, clone_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import Callback

from var4 import gen_data


class SaveBestCallback(Callback):

    MODEL_DUMP_TEMPLATE = "{date}{prefix}{number}"

    def __init__(self, prefix="best_model") -> None:
        super().__init__()
        self.prefix = prefix
        self.top_results = [-1, -1, -1]
        self.models = [None, None, None]

    def on_epoch_end(self, epoch: int, logs=None):
        result = logs["val_accuracy"]
        if result < min(self.top_results):
            return

        for i in range(3):
            if self.top_results[i] < result:
                for j in range(2, -1, -1):
                    self.top_results[j] = self.top_results[j - 1]
                    if self.models[j - 1]:
                        self.models[j] = clone_model(self.models[j - 1])

                self.top_results[i] = result
                self.models[i] = clone_model(self.model)
                break

    def on_train_end(self, logs=None):
        print(f"top results = {self.top_results}")
        print(f"models = {self.models}")
        date = datetime.now()
        for number in range(3):
            self.models[number].save(
                SaveBestCallback.MODEL_DUMP_TEMPLATE.format(
                    date=date, prefix=self.prefix, number=number
                )
            )


def main():
    batch_size = 32  # in each iteration, we consider 32 training examples at once
    num_epochs = 20  # we iterate 200 times over the entire training set
    kernel_size = 10  # we will use 3x3 kernels throughout
    pool_size = 5  # we will use 2x2 pooling throughout
    conv_depth_1 = 32  # we will initially have 32 kernels per conv. layer...
    drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5  # dropout in the dense layer with probability 0.5
    hidden_size = 20  # the dense layer will have 512 neurons

    data, raw_labels = gen_data()
    label_encoder = LabelEncoder()
    label_encoder.fit(np.unique(raw_labels))
    labels = label_encoder.transform(raw_labels)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

    _, height, width = train_data.shape

    inp = Input(shape=(height, width, 1))
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    conv_1 = Convolution2D(
        conv_depth_1, (kernel_size, kernel_size), padding="same", activation="relu"
    )(inp)
    conv_2 = Convolution2D(
        conv_depth_1, (kernel_size, kernel_size), padding="same", activation="relu"
    )(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)

    # Now flatten to 1D, apply Dense -> ReLU (with dropout) -> softmax
    flat = Flatten()(drop_1)
    hidden = Dense(hidden_size, activation="relu")(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(1, activation="sigmoid")(drop_3)
    model = Model(
        inputs=inp, outputs=out
    )  # To define a model, just specify its input and output layers

    model.compile(
        loss="binary_crossentropy",  # using the cross-entropy loss function
        optimizer="adam",  # using the Adam optimiser
        metrics=["accuracy"],
    )  # reporting the accuracy

    save_best_callback = SaveBestCallback()

    model.fit(
        train_data,
        train_labels,  # Train the model using the training set...
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[save_best_callback],
    )  # ...holding out 10% of the data for validation
    model.evaluate(test_data, test_labels, verbose=1)  # Evaluate the trained model on the test set!


if __name__ == "__main__":
    main()
