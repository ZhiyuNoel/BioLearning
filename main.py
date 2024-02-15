import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow
from src import mainWindow


class mainPage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = mainWindow.Ui_MainWindow(self)  # 初始化UI
        self.ui.click_bind()  # 假设这是绑定信号和槽的方法

    def closeEvent(self, event):
        self.ui.close()
        super().closeEvent(event)  # 确保调用父类的closeEvent


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("Images/BioLearning_Logo.png"))

    # 使用自定义的 QMainWindow 类
    mainpage = mainPage()
    mainpage.setFixedSize(1000, 820)
    mainpage.setWindowTitle("BioLearning 2.0")
    mainpage.show()
    sys.exit(app.exec())

