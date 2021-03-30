from PyQt5 import QtWidgets, QtGui, QtCore


WHITE = QtGui.QColor(255, 255, 255)
BLACK = QtGui.QColor(0, 0, 0)
PM_WIDTH = 420


class Canvas(QtWidgets.QLabel):
    def __init__(self, parent):
        super().__init__()
        self._parent = parent
        pixmap = QtGui.QPixmap(PM_WIDTH, PM_WIDTH)
        pixmap.fill(BLACK)
        self.setPixmap(pixmap)
        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        painter = QtGui.QPainter(self.pixmap())
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p = painter.pen()
        radius = 10
        p.setWidth(2 * radius)
        p.setColor(WHITE)
        painter.setPen(p)
        painter.drawEllipse(QtCore.QPoint(self.last_x, self.last_y), radius, radius)
        painter.end()
        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self._parent._update_results()
        self.last_x = None
        self.last_y = None

    def erase(self):
        self.pixmap().fill(BLACK)
        self.update()

    def load(self, filename):
        image = QtGui.QImage(filename)
        image = image.convertToFormat(QtGui.QImage.Format_Grayscale8)
        pm = QtGui.QPixmap(image)
        pm = pm.scaled(420, 420)

        self.setPixmap(pm)
        self.update()

