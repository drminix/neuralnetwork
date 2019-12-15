# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui\catclassifiermainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(903, 600)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 871, 71))
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 30, 841, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit_targetDirectory = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_targetDirectory.setObjectName("lineEdit_targetDirectory")
        self.horizontalLayout.addWidget(self.lineEdit_targetDirectory)
        self.pushButton_targetDirectory = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_targetDirectory.setObjectName("pushButton_targetDirectory")
        self.horizontalLayout.addWidget(self.pushButton_targetDirectory)
        self.widget_peformance = QtWidgets.QWidget(self.centralwidget)
        self.widget_peformance.setGeometry(QtCore.QRect(20, 130, 551, 411))
        self.widget_peformance.setObjectName("widget_peformance")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 110, 151, 16))
        self.label_2.setObjectName("label_2")
        self.textEdit_console = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_console.setGeometry(QtCore.QRect(580, 130, 311, 411))
        self.textEdit_console.setObjectName("textEdit_console")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(580, 110, 151, 16))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 903, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.label.setText(_translate("MainWindow", "Target Directory:"))
        self.pushButton_targetDirectory.setText(_translate("MainWindow", "Browse.."))
        self.label_2.setText(_translate("MainWindow", "Performance"))
        self.label_3.setText(_translate("MainWindow", "Console"))
