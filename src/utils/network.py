import sys
import socket
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel


class TrainingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.label = QLabel('Enter training parameters:')
        self.layout.addWidget(self.label)

        self.param1_input = QLineEdit(self)
        self.param1_input.setPlaceholderText('Parameter 1')
        self.layout.addWidget(self.param1_input)

        self.param2_input = QLineEdit(self)
        self.param2_input.setPlaceholderText('Parameter 2')
        self.layout.addWidget(self.param2_input)

        self.train_button = QPushButton('Start Training', self)
        self.train_button.clicked.connect(self.start_training)
        self.layout.addWidget(self.train_button)

        self.result_text = QTextEdit(self)
        self.layout.addWidget(self.result_text)

        self.setLayout(self.layout)
        self.setWindowTitle('ML Training App')
        self.show()

    def start_training(self):
        param1 = self.param1_input.text()
        param2 = self.param2_input.text()

        data = f"{param1},{param2}"

        # 创建 socket 对象
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 获取本地主机名
        host = '<Windows/Linux_IP>'
        port = 9999

        # 连接服务，指定主机和端口
        client_socket.connect((host, port))

        # 发送数据
        client_socket.send(data.encode())

        # 接收数据
        response = client_socket.recv(4096).decode()
        self.result_text.setText(response)

        # 关闭连接
        client_socket.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TrainingApp()
    sys.exit(app.exec())
