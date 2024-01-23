from abc import abstractmethod

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget


class Pages(QWidget):
    _signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()

    @abstractmethod
    def retranslateUi(self, Form):
        pass


