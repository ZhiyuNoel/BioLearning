import sys

from PyQt6.QtWidgets import QApplication, QMainWindow
from src import mainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 创建一个窗口
    mainpage = QMainWindow()
    mainui = mainWindow.Ui_MainWindow(mainpage)
        # 显示窗口
    mainui.click_bind()
    mainpage.setFixedSize(1000, 820)
    mainpage.show()    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec())

# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
#
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.initUI()
#
#     def initUI(self):
#         self.setWindowTitle("文件夹选择")
#         self.setGeometry(100, 100, 300, 200)
#
#         button = QPushButton("选择文件夹", self)
#         button.clicked.connect(self.slot_chooseDir)
#         button.setGeometry(100, 80, 100, 30)
#
#     def slot_chooseDir(self):  # 槽函数
#         fileName, fileType = QFileDialog.getOpenFileName(self, "选取文件",
#                                                                        "All Files(*);;Text Files(*.txt)")
#         print(fileName)
#         print(fileType)
#
#         if fileName == "":
#             print("\n取消选择")
#             return
#
#         print("\n你选择的文件为:")
#         print(fileName)
#         print("文件筛选器类型: ", fileType)
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     mainWindow = MainWindow()
#     mainWindow.show()
#     sys.exit(app.exec_())