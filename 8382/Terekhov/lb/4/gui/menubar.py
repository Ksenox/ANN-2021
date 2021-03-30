from PyQt5 import QtWidgets, QtCore


class MenuBar(QtWidgets.QMenuBar):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent

        self._main_menu = QtWidgets.QMenu("Menu")
        self._erase_action = QtWidgets.QAction("Erase")
        self._save_action = QtWidgets.QAction("Save")
        self._open_action = QtWidgets.QAction("Open")

        self._erase_action.setShortcut('Ctrl+X')
        self._erase_action.triggered.connect(self._parent.erase)

        self._save_action.setShortcut('Ctrl+S')
        self._save_action.triggered.connect(self._parent.save)

        self._open_action.setShortcut('Ctrl+O')
        self._open_action.triggered.connect(self._parent.open)

        self._main_menu.addAction(self._erase_action)
        self._main_menu.addAction(self._save_action)
        self._main_menu.addAction(self._open_action)
        self.addMenu(self._main_menu)




