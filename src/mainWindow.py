# Form implementation generated from reading ui file '../UI_files/mainWindow.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QMainWindow
from src.subpages import (Ui_inputpage, Ui_model, Ui_encoder,
                          Ui_pred, Ui_time, Ui_eveModel, Ui_evePre, Ui_eveEncoder, Ui_inter)


class Ui_MainWindow(QMainWindow):
    subpages = {}
    input_url = []
    pages = [Ui_inputpage, Ui_model, Ui_encoder, Ui_pred, Ui_time, Ui_eveModel,
             Ui_evePre, Ui_eveEncoder, Ui_inter]
    page_names = ["Ui_inputpage", "Ui_model", "Ui_encoder", "Ui_pred", "Ui_time", "Ui_eveModel",
                  "Ui_evePre", "Ui_eveEncoder", "Ui_inter"]
    _signal_url = pyqtSignal(list)

    def __init__(self, MainWindow):
        super().__init__()
        MainWindow.resize(1000, 800)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 980, 780))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.buttonLayout = QtWidgets.QVBoxLayout()
        self.buttonLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.buttonLayout.setSpacing(20)
        self.buttonLayout.setObjectName("verticalLayout")

        ## Select input button
        self.inputButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.inputButton.setMinimumSize(QtCore.QSize(0, 50))
        self.inputButton.setObjectName("pushButton")
        self.buttonLayout.addWidget(self.inputButton)

        ## Model Selection Button
        self.modelButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.modelButton.setMinimumSize(QtCore.QSize(0, 50))
        self.modelButton.setObjectName("pushButton_4")
        self.buttonLayout.addWidget(self.modelButton)

        ## Train autoencoder button
        self.autoencodeButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.autoencodeButton.setMinimumSize(QtCore.QSize(0, 50))
        self.autoencodeButton.setObjectName("pushButton_2")
        self.buttonLayout.addWidget(self.autoencodeButton)

        ## Train prediction button
        self.predictorButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.predictorButton.setMinimumSize(QtCore.QSize(0, 50))
        self.predictorButton.setObjectName("pushButton_5")
        self.buttonLayout.addWidget(self.predictorButton)

        ## Event time button
        self.timeButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.timeButton.setMinimumSize(QtCore.QSize(0, 50))
        self.timeButton.setObjectName("pushButton_6")
        self.buttonLayout.addWidget(self.timeButton)

        ## Event Model selection button
        self.eventModelButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.eventModelButton.setMinimumSize(QtCore.QSize(0, 50))
        self.eventModelButton.setObjectName("pushButton_7")
        self.buttonLayout.addWidget(self.eventModelButton)

        ## Event predictor button
        self.eventPredictorButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.eventPredictorButton.setMinimumSize(QtCore.QSize(0, 50))
        self.eventPredictorButton.setObjectName("pushButton_8")
        self.buttonLayout.addWidget(self.eventPredictorButton)

        ## Event Autoencoder button
        self.eventEncoderButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.eventEncoderButton.setMinimumSize(QtCore.QSize(0, 50))
        self.eventEncoderButton.setObjectName("pushButton_9")
        self.buttonLayout.addWidget(self.eventEncoderButton)

        ## Interpretation button
        self.interpretationButton = QtWidgets.QPushButton(parent=self.verticalLayoutWidget_2)
        self.interpretationButton.setEnabled(True)
        self.interpretationButton.setMinimumSize(QtCore.QSize(0, 50))
        self.interpretationButton.setObjectName("pushButton_10")
        self.buttonLayout.addWidget(self.interpretationButton)

        policy_subpage = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                                               QtWidgets.QSizePolicy.Policy.Minimum)
        policy_subpage.setHorizontalStretch(0)
        policy_subpage.setVerticalStretch(0)

        self.horizontalLayout_2.addLayout(self.buttonLayout)
        self.subpage = QtWidgets.QStackedWidget(parent=self.verticalLayoutWidget_2)
        self.subpage.setObjectName("stackedWidget")
        self.subpage.setSizePolicy(policy_subpage)

        MainWindow.setCentralWidget(self.centralwidget)
        self.horizontalLayout_2.addWidget(self.subpage)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.textBrowser = QtWidgets.QTextBrowser(parent=self.verticalLayoutWidget_2)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_2.addWidget(self.textBrowser)
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.stack_fill()
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.inputButton.setText(_translate("MainWindow", "Select Input"))
        self.modelButton.setText(_translate("MainWindow", "Select Model"))
        self.autoencodeButton.setText(_translate("MainWindow", "Train Autoencoder"))
        self.predictorButton.setText(_translate("MainWindow", "Train Predictor"))
        self.timeButton.setText(_translate("MainWindow", "Event Time"))
        self.eventModelButton.setText(_translate("MainWindow", "Select Event Model"))
        self.eventPredictorButton.setText(_translate("MainWindow", "Event Predictor"))
        self.eventEncoderButton.setText(_translate("MainWindow", "Event Autoencoder"))
        self.interpretationButton.setText(_translate("MainWindow", "Interpretation"))

    def click_bind(self):
        self.predictorButton.clicked.connect(lambda: self.print_in_textBrowser("Click the train prediction button"))
        self.autoencodeButton.clicked.connect(lambda: self.print_in_textBrowser("Click the train autoencoder button"))
        self.autoencodeButton.clicked.connect(lambda: self.send_url())
        self.modelButton.clicked.connect(lambda: self.print_in_textBrowser("Click the select model button"))
        self.inputButton.clicked.connect(lambda: self.print_in_textBrowser("Click the select input button"))
        self.timeButton.clicked.connect(lambda: self.print_in_textBrowser("Click the select time button"))
        self.eventModelButton.clicked.connect(lambda: self.print_in_textBrowser("Click the select event model button"))
        self.eventPredictorButton.clicked.connect(
            lambda: self.print_in_textBrowser("Click the select event predictor button"))
        self.eventEncoderButton.clicked.connect(
            lambda: self.print_in_textBrowser("Click the select event encoder button"))
        self.interpretationButton.clicked.connect(
            lambda: self.print_in_textBrowser("Click the select interpretation button"))

        self.inputButton.clicked.connect(lambda: self.page_switch_clicked(0))
        self.modelButton.clicked.connect(lambda: self.page_switch_clicked(1))
        self.autoencodeButton.clicked.connect(lambda: self.page_switch_clicked(2))
        self.predictorButton.clicked.connect(lambda: self.page_switch_clicked(3))
        self.timeButton.clicked.connect(lambda: self.page_switch_clicked(4))
        self.eventModelButton.clicked.connect(lambda: self.page_switch_clicked(5))
        self.eventPredictorButton.clicked.connect(lambda: self.page_switch_clicked(6))
        self.eventEncoderButton.clicked.connect(lambda: self.page_switch_clicked(7))
        self.interpretationButton.clicked.connect(lambda: self.page_switch_clicked(8))

    def stack_fill(self):
        # Check if the lists have the same length
        if len(self.pages) != len(self.page_names):
            raise ValueError("pages and page_names must be of the same length")

        for page, page_name in zip(self.pages, self.page_names):
            try:
                insertpage = QtWidgets.QWidget()
                Page_ui = page(insertpage)
                Page_ui._signal.connect(self.print_in_textBrowser)
                self.subpage.addWidget(insertpage)
                self.subpages[page_name] = Page_ui
            except Exception as e:
                # Handle exceptions (log or take appropriate action)
                print(f"Error loading page {page_name}: {e}")

        self.additional_connect()


    def additional_connect(self):
        self.subpages["Ui_inputpage"]._signal_list.connect(self.load_input_url)
        self._signal_url.connect(self.subpages["Ui_encoder"].receive_url)

    def send_url(self):
        self._signal_url.emit(self.input_url)

    def load_input_url(self, url_dic):
        self.input_url = url_dic


    def print_in_textBrowser(self, text):
        self.textBrowser.setText(text)

    ## 10 pages in total
    def page_switch_clicked(self, index):
        self.subpage.setCurrentIndex(index)
