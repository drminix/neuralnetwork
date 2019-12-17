#standard modules
import os
import sys

#third party modules
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QAction
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import pyqtSignal, QObject
import skimage.io as io
import skimage.transform as trans

#local modules
from gui import ui_catclassifiermainwindow

class DataProcessor(QObject):

    #class variable
    TEST_DIRECTORY = "test"
    TRAIN_DIRECTORY = "train"
    WIDTH = HEIGHT = 256

    #signals
    fireData = pyqtSignal([str])

    def __init__(self,target_directory: str):
        print(type(target_directory))
        super().__init__()
        self._target_directory = target_directory
        self._train_directory = os.path.join(self._target_directory, DataProcessor.TRAIN_DIRECTORY)
        self._test_directory = os.path.join(self._target_directory, DataProcessor.TEST_DIRECTORY)

    def load_train_data(self):
        X = none

        for file in os.scandir(self._train_directory):
            self.fireData.emit(file.path)
            #load data
            current_x =

class CatClassifierMainWindow(QMainWindow):
    def __init__(self):
        super().__init__() #call parent's constructor

        #(1) setup basic UI
        self._ui = ui_catclassifiermainwindow.Ui_MainWindow()
        self._ui.setupUi(self)
        self.setWindowTitle("CatClassification Neural Network Training Tool")
        self.setFixedSize(self.size())
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        #(2) setup events
        self._setup_events()

    def _setup_events(self):
        self._ui.pushButton_targetDirectory.clicked.connect(self._show_dialog_target_directory)

    def _show_dialog_target_directory(self):
        dirname = QFileDialog.getExistingDirectory (self, "Select target directory which contains all the files")
        if dirname:
            self._ui.lineEdit_targetDirectory.setText(dirname)
            self._dataprocessor = DataProcessor(dirname)
            self._dataprocessor.fireData.connect(self.update_console)
            self._dataprocessor.load_train_data()

    def update_console(self, line:str):
        self._ui.textEdit_console.append(line)

if __name__ == "__main__":

     app = QApplication(sys.argv)

     classifier = CatClassifierMainWindow()
     classifier.show()

     sys.exit(app.exec_())
