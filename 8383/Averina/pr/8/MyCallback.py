from datetime import datetime

from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import clone_model


class MyCallback(Callback):

    def __init__(self, prefix="MODEL"):
        super().__init__()
        self.m = 3
        self.top = [-1] * self.m
        self.models = [None] * self.m
        self.numbers = [-1] * self.m
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        acc = logs["val_accuracy"]
        if acc < min(self.top):
            return

        for i in range(self.m):
            if acc >= self.top[i] and epoch not in self.numbers:

                for j in range(self.m - 2, -1, -1):

                    self.top[j + 1] = self.top[j]
                    self.numbers[j + 1] = self.numbers[j]
                    if self.models[j]:
                        self.models[j + 1] = clone_model(self.models[j])
                self.top[i] = acc
                self.models[i] = clone_model(self.model)
                self.numbers[i] = epoch
                break

    def on_train_end(self, logs=None):
        print(f"Results: {self.top}")
        print(f"Numbers: {self.numbers}")
        print(f"Models: {self.models}")
        date = datetime.now()
        for i in range(self.m):
            self.models[i].save(f"{date}_{self.prefix}_{self.numbers[i]}")
