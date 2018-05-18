# -*- coding: utf-8 -*-
import sys

from PyQt5 import QtGui,QtWidgets


def show_image(image_path='equation.jpeg'):
    app = QtWidgets.QApplication(sys.argv)
    pixmap = QtGui.QPixmap(image_path)
    screen = QtWidgets.QLabel()
    screen.setPixmap(pixmap)
    screen.showFullScreen()
    sys.exit(app.exec_())


if __name__ == '__main__':
    show_image()
