import numpy as np

from models.model_handler import ModelHandler
from gui.canvas import Canvas, PM_WIDTH
from gui.menubar import MenuBar

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QRect


WIN_SIZE = (600, 490)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._model_handler = ModelHandler("model_1.h5")
        self.setWindowTitle("ANN #4")
        self.setFixedSize(*WIN_SIZE)
        self._canvas = Canvas(self)
        self._menubar = MenuBar(self)

        self._labels = [QtWidgets.QLabel(f'{x}: 0.0') for x in range(10)]
        self._result_label = QtWidgets.QLabel(f'It`s ..?')
        self._result_label.setFont(QtGui.QFont('Arial', 16))

        self._right_layout = QtWidgets.QVBoxLayout()
        for label in self._labels:
            self._right_layout.addWidget(label)
        self._right_layout.addWidget(self._result_label)

        self._main_layout = QtWidgets.QHBoxLayout()
        self._main_layout.addWidget(self._canvas)
        self._main_layout.addLayout(self._right_layout)
        self.setLayout(self._main_layout)

    def save(self):
        p = self._canvas.grab(QRect(0, 24, PM_WIDTH, PM_WIDTH))
        p = p.scaled(28, 28, transformMode=QtCore.Qt.SmoothTransformation)
        fileName = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '')
        with open(fileName[0], 'w'):
            p.save(fileName[0], "PNG")

    def erase(self):
        self._canvas.erase()
        for i in range(len(self._labels)):
            self._labels[i].setText(f'{i}: 0.0')
        self._result_label.setText(f'It`s ..?')

    def open(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '')
        if len(fileName[0]) != 0:
            self._canvas.load(fileName[0])
            self._update_results()

    def _update_results(self):
        pic = self._canvas.grab(QRect(0, 0, PM_WIDTH, PM_WIDTH))
        pic = pic.scaled(28, 28, transformMode=QtCore.Qt.SmoothTransformation)
        pic = self._pixmap_to_array(pic)
        prediction, digit = self._get_prediction(pic)
        for i in range(len(prediction)):
            self._labels[i].setText(f'{i}: {prediction[i]}')
        if float(prediction[digit]) > .80:
            self._result_label.setText(f'It`s {digit}!')
        else:
            self._result_label.setText(f'It`s ..?')

    def _get_prediction(self, array):
        result = self._model_handler._model.predict(array)
        maximum = np.argmax(result, axis=1)
        result = ["{:0.4f}".format(v) for v in result[0]]
        return result, maximum[0]

    @staticmethod
    def _pixmap_to_array(pixmap):
        image = pixmap.toImage()
        b = image.bits()
        b.setsize(28 * 28 * 4)
        arr = np.frombuffer(b, np.uint8).reshape((28, 28, 4))
        ret = np.array([[x[0] for x in line] for line in arr])
        ret[0] = np.array([0 for _ in range(28)])
        ret = ret / 255.0
        return np.array([ret])
