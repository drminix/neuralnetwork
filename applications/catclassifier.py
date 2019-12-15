#standard modules
import os
import sys

#third party modules
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication

#local modules

from gui import ui_catclassifiermainwindow

class DataProcessor:
    TEST_DIRECTORY = "test"
    TRAIN_DIRECTORY = "train"

    def __init__(self,target_directory: str):
        self._target_directory = target_directory
        self._train_directory = os.path.join(self._target_directory, DataProcessor.TEST_DIRECTORY)
        self._test_directory = os.path.join(self._target_directory, DataProcessor.TEST_DIRECTORY)

        if not os.path.exists(self._train_directory) or not os.path.exists(self._test_directory):
            pass

class CatClassifierMainWindow(QMainWindow):
    def __init__(self):
        super().__init__() #call parent's constructor

        #(1) setup basic UI
        self._ui = ui_catclassifiermainwindow.Ui_MainWindow()
        self._ui.setupUi(self)

if __name__ == "__main__":

     app = QApplication(sys.argv)

     classifier = CatClassifierMainWindow()
     classifier.show()

     sys.exit(app.exec_())
