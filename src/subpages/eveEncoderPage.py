# Form implementation generated from reading ui file '../UI_files/inputPage.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtWidgets
from .Pages import Pages

class Ui_eveEncoder(Pages):
    def __init__(self, inpPage):
        super().__init__()
        inpPage.setObjectName("Input Page")
        inpPage.resize(611, 531)
        self.textEdit = QtWidgets.QTextEdit(parent=inpPage)
        self.textEdit.setGeometry(QtCore.QRect(10, 10, 81, 31))
        self.textEdit.setObjectName("textEdit")
        self.retranslateUi(inpPage)
        QtCore.QMetaObject.connectSlotsByName(inpPage)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.textEdit.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
        "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
        "p, li { white-space: pre-wrap; }\n"
        "hr { height: 1px; border-width: 0; }\n"
        "li.unchecked::marker { content: \"\\2610\"; }\n"
        "li.checked::marker { content: \"\\2612\"; }\n"
        "</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
        "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">event autoencoder page</p></body></html>"))
