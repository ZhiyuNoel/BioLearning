# Form implementation generated from reading ui file '../UI_files/inputPage.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPixmap

from .Pages import Pages
from ..model import TrainingThread, LinearAutoencoder
from ..model.predict import TestThread
from ..utils import loader_pipeline


class Ui_encoder(Pages):
    start_lr = 0
    end_lr = 0
    batch_size = 0
    window_size = 0
    input_url = []
    model_info = {}
    trainLoader = None
    testLoader = None

    def __init__(self, encoPage):
        super().__init__()
        self.training_thread = None
        encoPage.setObjectName("Form")
        encoPage.resize(820, 531)
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=encoPage)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 820, 520))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.verticalLayout.setContentsMargins(5, 5, 0, 5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.imgWinLayout = QtWidgets.QHBoxLayout()
        self.imgWinLayout.setObjectName("horizontalLayout")

        self.orgImgWin = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.orgImgWin.setMinimumSize(QtCore.QSize(300, 300))
        self.orgImgWin.setObjectName("widget")
        self.imgWinLayout.addWidget(self.orgImgWin)

        self.recImgWin = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        self.recImgWin.setMinimumSize(QtCore.QSize(300, 300))
        self.recImgWin.setObjectName("widget_2")
        self.imgWinLayout.addWidget(self.recImgWin)

        self.verticalLayout.addLayout(self.imgWinLayout)
        self.terminalLayout = QtWidgets.QHBoxLayout()
        self.terminalLayout.setSpacing(10)
        self.terminalLayout.setObjectName("horizontalLayout_2")

        self.operatorLayout = QtWidgets.QVBoxLayout()
        self.operatorLayout.setObjectName("verticalLayout_2")

        self.parameterLayout = QtWidgets.QVBoxLayout()
        self.parameterLayout.setSpacing(10)
        self.parameterLayout.setObjectName("verticalLayout_3")

        self.startLrLayout = QtWidgets.QHBoxLayout()
        self.startLrLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.startLrLayout.setSpacing(0)
        self.startLrLayout.setObjectName("horizontalLayout_4")
        self.text_startLr = QtWidgets.QTextBrowser(parent=self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.text_startLr.sizePolicy().hasHeightForWidth())
        self.text_startLr.setSizePolicy(sizePolicy)
        self.text_startLr.setMaximumSize(QtCore.QSize(200, 30))
        self.text_startLr.setObjectName("textBrowser_2")
        self.startLrLayout.addWidget(self.text_startLr)

        self.startLrSpin = QtWidgets.QDoubleSpinBox(parent=self.verticalLayoutWidget)
        self.startLrSpin.setMinimumSize(QtCore.QSize(200, 30))
        self.startLrSpin.setObjectName("doubleSpinBox")
        self.startLrSpin.setDecimals(8)
        self.startLrSpin.setValue(0.001)
        self.startLrSpin.setSingleStep(0.00001)
        self.startLrLayout.addWidget(self.startLrSpin)
        self.parameterLayout.addLayout(self.startLrLayout)

        self.endLrLayout = QtWidgets.QHBoxLayout()
        self.endLrLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.endLrLayout.setSpacing(0)
        self.endLrLayout.setObjectName("horizontalLayout_8")
        self.text_endLr = QtWidgets.QTextBrowser(parent=self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        sizePolicy.setHeightForWidth(self.text_endLr.sizePolicy().hasHeightForWidth())
        self.text_endLr.setSizePolicy(sizePolicy)
        self.text_endLr.setMaximumSize(QtCore.QSize(200, 30))
        self.text_endLr.setObjectName("textBrowser_3")

        self.endLrLayout.addWidget(self.text_endLr)
        self.endLrSpin = QtWidgets.QDoubleSpinBox(parent=self.verticalLayoutWidget)
        self.endLrSpin.setMinimumSize(QtCore.QSize(200, 30))
        self.endLrSpin.setObjectName("doubleSpinBox_2")
        self.endLrSpin.setDecimals(8)
        self.endLrSpin.setValue(0.0001)
        self.endLrSpin.setSingleStep(0.00001)
        self.endLrLayout.addWidget(self.endLrSpin)
        self.parameterLayout.addLayout(self.endLrLayout)

        self.wsLayout = QtWidgets.QHBoxLayout()
        self.wsLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.wsLayout.setSpacing(0)
        self.wsLayout.setObjectName("horizontalLayout_9")
        self.text_ws = QtWidgets.QTextBrowser(parent=self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        sizePolicy.setHeightForWidth(self.text_ws.sizePolicy().hasHeightForWidth())
        self.text_ws.setSizePolicy(sizePolicy)
        self.text_ws.setMaximumSize(QtCore.QSize(200, 30))
        self.text_ws.setObjectName("textBrowser_6")

        self.wsLayout.addWidget(self.text_ws)
        self.wsSpin = QtWidgets.QSpinBox(parent=self.verticalLayoutWidget)
        self.wsSpin.setMinimumSize(QtCore.QSize(200, 30))
        self.wsSpin.setObjectName("doubleSpinBox_4")
        self.wsSpin.setMaximum(1000)
        self.wsSpin.setMinimum(1)
        self.wsSpin.setSingleStep(1)
        self.wsSpin.setValue(100)
        self.wsLayout.addWidget(self.wsSpin)
        self.parameterLayout.addLayout(self.wsLayout)

        self.bsLayout = QtWidgets.QHBoxLayout()
        self.bsLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.bsLayout.setSpacing(0)
        self.bsLayout.setObjectName("horizontalLayout_5")
        self.text_bs = QtWidgets.QTextBrowser(parent=self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        sizePolicy.setHeightForWidth(self.text_bs.sizePolicy().hasHeightForWidth())
        self.text_bs.setSizePolicy(sizePolicy)
        self.text_bs.setMaximumSize(QtCore.QSize(200, 30))
        self.text_bs.setObjectName("textBrowser_4")
        self.bsLayout.addWidget(self.text_bs)
        self.bsSpin = QtWidgets.QSpinBox(parent=self.verticalLayoutWidget)
        self.bsSpin.setMinimumSize(QtCore.QSize(150, 30))
        self.bsSpin.setObjectName("doubleSpinBox_3")
        self.bsSpin.setMaximum(100)
        self.bsSpin.setMinimum(1)
        self.bsSpin.setSingleStep(1)
        self.bsSpin.setValue(10)
        self.bsLayout.addWidget(self.bsSpin)
        self.parameterLayout.addLayout(self.bsLayout)

        self.operatorLayout.addLayout(self.parameterLayout)
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.buttonLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        self.buttonLayout.setContentsMargins(0, 5, 0, 5)
        self.buttonLayout.setSpacing(5)
        self.buttonLayout.setObjectName("horizontalLayout_3")

        buttonSizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                                                 QtWidgets.QSizePolicy.Policy.Minimum)
        self.confirmButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget)
        self.confirmButton.setObjectName("pushButton")
        self.confirmButton.setSizePolicy(buttonSizePolicy)
        self.buttonLayout.addWidget(self.confirmButton)

        self.trainButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget)
        self.trainButton.setObjectName("trainButton")
        self.trainButton.setSizePolicy(buttonSizePolicy)
        self.trainButton.setVisible(True)
        self.buttonLayout.addWidget(self.trainButton)

        self.stopTrain = QtWidgets.QPushButton(parent=self.verticalLayoutWidget)
        self.stopTrain.setObjectName("pushButton_4")
        self.stopTrain.setSizePolicy(buttonSizePolicy)
        self.stopTrain.setVisible(False)
        self.buttonLayout.addWidget(self.stopTrain)

        self.testButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget)
        self.testButton.setObjectName("pushButton")
        self.testButton.setSizePolicy(buttonSizePolicy)
        self.testButton.setVisible(True)
        self.buttonLayout.addWidget(self.testButton)

        self.pauseTest = QtWidgets.QPushButton(parent=self.verticalLayoutWidget)
        self.pauseTest.setObjectName("pauseTest")
        self.pauseTest.setSizePolicy(buttonSizePolicy)
        self.pauseTest.setVisible(False)
        self.buttonLayout.addWidget(self.pauseTest)

        self.clearButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget)
        self.clearButton.setObjectName("pushButton")
        self.clearButton.setSizePolicy(buttonSizePolicy)
        self.buttonLayout.addWidget(self.clearButton)
        self.operatorLayout.addLayout(self.buttonLayout)

        self.terminalLayout.addLayout(self.operatorLayout)

        self.showLayout = QtWidgets.QVBoxLayout()
        self.showLayout.setObjectName("showLayout")
        font = QFont()
        font.setBold(True)
        font.setWeight(30)
        self.processBar = QtWidgets.QProgressBar()
        self.processBar.setObjectName("processBar")
        self.processBar.setMinimum(0)
        self.processBar.setMinimumSize(QtCore.QSize(300, 0))
        self.processBar.setMaximumSize(QtCore.QSize(400, 16777215))
        self.processBar.setStyleSheet(
            "QProgressBar { border: 2px solid grey; border-radius: 5px; color: rgb(20,20,20);  background-color: "
            "#FFFFFF; text-align: center;}QProgressBar::chunk {background-color: rgb(100,200,200); border-radius: "
            "10px; margin: 0.1px;  width: 1px;}")
        self.processBar.setFont(font)
        self.showLayout.addWidget(self.processBar)

        self.textBrowser = QtWidgets.QTextBrowser(parent=self.verticalLayoutWidget)
        self.textBrowser.setEnabled(True)
        self.textBrowser.setMinimumSize(QtCore.QSize(300, 0))
        self.textBrowser.setMaximumSize(QtCore.QSize(400, 16777215))
        self.textBrowser.setReadOnly(True)
        self.textBrowser.setObjectName("textBrowser")

        self.showLayout.addWidget(self.textBrowser)
        self.terminalLayout.addLayout(self.showLayout)
        self.verticalLayout.addLayout(self.terminalLayout)
        self.retranslateUi(encoPage)
        QtCore.QMetaObject.connectSlotsByName(encoPage)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.text_startLr.setText("Start Learning Rate")
        self.text_endLr.setText("End Learning Rate")
        self.text_bs.setText("Batch Size")
        self.text_ws.setText("Slide Window Size")

        self.clearButton.setText(_translate("Form", "Clear"))
        self.trainButton.setText(_translate("Form", "Train"))
        self.testButton.setText(_translate("Form", "Test"))
        self.confirmButton.setText(_translate("Form", "Confirm Parameters"))
        self.stopTrain.setText(_translate("Form", "Pause"))
        self.pauseTest.setText(_translate("Form", "Pause"))

        self.trainButton.setEnabled(False)
        self.testButton.setEnabled(False)
        self.clearButton.setEnabled(True)

        self.confirmButton.clicked.connect(self.confirm_parameter)
        self.clearButton.clicked.connect(self.clear_parameter)

        self.trainButton.clicked.connect(self.start_train_pipeline)
        self.stopTrain.clicked.connect(self.pause_training)
        self.testButton.clicked.connect(self.start_test_pipeline)
        self.pauseTest.clicked.connect(self.pause_test)

    def confirm_parameter(self):
        if len(self.input_url) != 2:
            self.send_data("Invalid Dataset Input! Please Input Data Again")
            return
        self.trainButton.setEnabled(True)
        self.testButton.setEnabled(True)
        self.clearButton.setEnabled(True)
        self.confirmButton.setEnabled(False)

        self.bsSpin.setEnabled(False)
        self.wsSpin.setEnabled(False)
        self.startLrSpin.setEnabled(False)
        self.endLrSpin.setEnabled(False)

        self.window_size = self.wsSpin.value()
        self.batch_size = self.bsSpin.value()
        self.start_lr = self.startLrSpin.value()
        self.end_lr = self.endLrSpin.value()

        loader_kargs = {"train_video": self.input_url[0], "train_label": self.input_url[1],
                        "test_video": "", "test_label": "", "window_size": self.window_size,
                        "batch_size": self.batch_size, "win_stride": 2, "imgz": (25, 25)}
        trainLoader, testLoader = loader_pipeline(**loader_kargs)
        self.textBrowser.setText("Data Load Complete")
        self.model = LinearAutoencoder()
        self.training_thread = TrainingThread(self.verticalLayoutWidget, self.model, trainLoader, self.start_lr)
        self.test_thread = TestThread(self.verticalLayoutWidget, self.model, testLoader)
        self.processBar.setMaximum(len(trainLoader))

    def clear_parameter(self):
        self.trainButton.setEnabled(False)
        self.testButton.setEnabled(False)
        self.confirmButton.setEnabled(True)
        self.clearButton.setEnabled(True)

        self.bsSpin.setEnabled(True)
        self.wsSpin.setEnabled(True)
        self.startLrSpin.setEnabled(True)
        self.endLrSpin.setEnabled(True)

        self.startLrSpin.setValue(0.0001)
        self.endLrSpin.setValue(0.0001)
        self.wsSpin.setValue(100)
        self.bsSpin.setValue(10)

        self.stop_training()

    def update_image(self, origin: QImage, result: QImage):
        pixmap_original = QPixmap.fromImage(origin)
        pixmap_reconstructed = QPixmap.fromImage(result)

        # 在标签中设置图像
        self.orgImgWin.setPixmap(pixmap_original)
        self.recImgWin.setPixmap(pixmap_reconstructed)

        # 适应标签大小
        self.orgImgWin.setScaledContents(True)
        self.recImgWin.setScaledContents(True)

    def send_data(self, str_data):
        self._signal.emit(str_data)

    def receive_url(self, input_url):
        self.input_url = input_url
        print(self.input_url)

    def receive_model(self, model_dict):
        self.model_info = model_dict
        print(self.model_info)

    def start_train_pipeline(self):
        self.pause_test()
        if not self.training_thread.isRunning():
            self.training_thread.update_progress.connect(self.update_progress_bar)
            self.training_thread.update_text.connect(self.update_text_browser)
            self.training_thread.update_pic.connect(self.update_image)
            self.training_thread.start()
        elif self.training_thread.paused:
            self.training_thread.resume()
        self.trainButton.setVisible(False)
        self.stopTrain.setVisible(True)


    def stop_training(self):
        if self.training_thread is None:
            return
        self.training_thread.stop()
        self.trainButton.setVisible(True)
        self.stopTrain.setVisible(False)

    def pause_training(self):
        if self.training_thread.isRunning():
            self.training_thread.pause()
        self.trainButton.setVisible(True)
        self.stopTrain.setVisible(False)
        self.model = self.training_thread.get_model()

    def start_test_pipeline(self):
        self.pause_training()
        self.test_thread.update_model(self.model)
        if not self.test_thread.isRunning():
            # self.test_thread.update_progress.connect(self.update_progress_bar)
            # self.test_thread.update_text.connect(self.update_text_browser)
            self.test_thread.update_pic.connect(self.update_image)
            self.test_thread.start()
        elif self.test_thread.paused:
            self.test_thread.resume()
        self.testButton.setVisible(False)
        self.pauseTest.setVisible(True)

    def pause_test(self):
        if self.test_thread.isRunning():
            self.test_thread.pause()
        self.testButton.setVisible(True)
        self.pauseTest.setVisible(False)

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.processBar.setValue(value)

    @pyqtSlot(str)
    def update_text_browser(self, text):
        self.textBrowser.setText(text)
