import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication
from tensorflow.keras import optimizers

from gui.main_window import MainWindow
from models.model_handler import ModelHandler

if __name__ == '__main__':
    if sys.argv[-1] == 'gui':
        app = QApplication(sys.argv)
        ui = MainWindow()
        ui.show()
        sys.exit(app.exec_())
    else:
        configs = [
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.Adam()},
            {"layers": [{"size": 256, "activation": 'relu'},
                        {"size": 64, "activation": 'relu'}], "optimizer": optimizers.Adam()},
            {"layers": [{"size": 128, "activation": 'relu'}], "optimizer": optimizers.Adam()},
            {"layers": [{"size": 64, "activation": 'relu'}], "optimizer": optimizers.Adam()},

            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.Adam(learning_rate=0.01)},
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.Adam(beta_1=0.1)},
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.Adam(beta_2=0.8)},

            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.Nadam()},
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.Nadam(learning_rate=0.01)},
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.Nadam(beta_1=0.1)},
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.Nadam(beta_2=0.8)},

            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.RMSprop()},
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.RMSprop(learning_rate=0.01)},
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.RMSprop(rho=0.5)},

            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.SGD()},
            {"layers": [{"size": 256, "activation": 'relu'}], "optimizer": optimizers.SGD(learning_rate=0.0001)},

        ]
        models = [ModelHandler(f"model_{i + 1}.h5", configs[i]) for i in range(len(configs))]
        with open("result.txt", "w") as f:
            for i in range(len(models)):
                df_hist = pd.DataFrame(models[i].get_history())
                f.write(f"ANN #{i + 1}\n")
                f.write(df_hist.to_string() + '\n')

