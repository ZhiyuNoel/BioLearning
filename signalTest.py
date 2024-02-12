# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:25:19 2021

@author: Wenqing Zhou (zhou.wenqing@gmail.com)
@github: https://github.com/ouening
"""

import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from random import randint


class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window.
    """
    _signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Another Window % d" % randint(0, 100))

        self.qline = QLineEdit("send_data")
        self.qline.textChanged.connect(self.send_data)

        self.btn = QPushButton()
        # self.btn.clicked.connect(self.send_data)# send data

        layout.addWidget(self.label)
        layout.addWidget(self.qline)
        layout.addWidget(self.btn)
        self.setLayout(layout)

    def send_data(self, str_data):
        # str_data = self.qline.text()
        print(str_data)

        self._signal.emit(str_data)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.w = AnotherWindow()
        self.button = QPushButton("Push for Window")
        self.button.clicked.connect(self.show_new_window)
        # self.button.clicked.connect(self.toggle_window)

        self.input = QLineEdit()
        self.input.textChanged.connect(self.w.label.setText)

        self.qlabel = QLabel('测试窗口通信')

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.input)
        layout.addWidget(self.qlabel)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def show_new_window(self, checked):
        self.w.show()
        self.w._signal.connect(self.process_data)

    def process_data(self, str_data):
        self.qlabel.setText(str_data)  # change qlabel text

    def toggle_window(self, checked):
        if self.w.isVisible():
            self.w.hide()
        else:
            self.w.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()
