import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow
from src import mainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("Images/BioLearning_Logo.png"))
    # 创建一个窗口
    mainpage = QMainWindow()
    mainui = mainWindow.Ui_MainWindow(mainpage)
        # 显示窗口
    mainui.click_bind()
    mainpage.setFixedSize(1000, 820)
    mainpage.setWindowTitle("BioLearning 2.0")

    mainpage.show()    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec())