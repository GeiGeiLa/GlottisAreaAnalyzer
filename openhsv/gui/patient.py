from PyQt5.QtWidgets import QApplication, QGridLayout, QPushButton, QLabel, \
    QMessageBox, QDialog, QLineEdit, QCheckBox, QDateEdit, QComboBox, QFileDialog
from PyQt5.QtGui import QIntValidator
import qdarkstyle
from os.path import isdir
import os
from glob import glob
import json

class Patient(QDialog):
    def __init__(self, base_folder):
        """Patient dialog

        :param base_folder: base folder where data will be saved
        :type base_folder: str
        """
        super().__init__()
        self.setWindowTitle("New patient")
        self.setFixedWidth(300)

        self.l = QGridLayout(self)

        self.combo = QComboBox()

        folders = [i.split("\\")[-1] for i in glob(base_folder+"\\*") if isdir(i)]

        if len(folders) == 0:
            os.mkdir(base_folder+"\\Sprechstunde")
            folders.append("Sprechstunde")

        self.combo.addItem("Sprechstunde")

        for i in folders:
            if i is not "Sprechstunde":
                self.combo.addItem(i)

        self.l.addWidget(QLabel("Folder"))
        self.l.addWidget(self.combo)

        self.l.addWidget(QLabel('Last Name / Identifier*'))
        self.last_name = QLineEdit()
        self.last_name.setPlaceholderText("e.g. Smith")
        self.l.addWidget(self.last_name)

        self.l.addWidget(QLabel("First Name"))
        self.first_name = QLineEdit()
        self.first_name.setPlaceholderText("e.g. John")
        self.l.addWidget(self.first_name)

        self.l.addWidget(QLabel("Birth data"))
        self.birth_date = QDateEdit()
        self.l.addWidget(self.birth_date)

        self.l.addWidget(QLabel("Comment"))
        self.comment = QLineEdit()
        self.comment.setPlaceholderText("e.g. RBH 022")
        self.l.addWidget(self.comment)

        b = QPushButton("Continue")
        b.clicked.connect(self.close)
        self.l.addWidget(b)

        self.l.addWidget(QLabel("*: mandatory fields"))

    def close(self):
        if not self.last_name.text():
            QMessageBox.critical(self, 
                "No identifier", 
                "Please enter identifier (e.g. last name) of the subject.")
            return

        super().close()

    def get(self):
        return dict(last_name=self.last_name.text(),
                    first_name=self.first_name.text(),
                    birth_date=self.birth_date.text(),
                    comment=self.comment.text(),
                    folder=self.combo.currentText())


if __name__ == '__main__':
    from PyQt5.QtWidgets import QWidget 
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    if os.path.exists("settings.json"):
        settings = json.load(open("settings.json"))
        baseFolder = settings['baseFolder']

    else:
        baseFolder = None
        
        while baseFolder is None:
            baseFolder = QFileDialog.getExistingDirectory()

    w = QWidget()

    QMessageBox.information(w, 
        "Dummy patient",
        "This is only for testing purposes and not for adding a patient to the database")

    p = Patient(base_folder=baseFolder)
    p.show()

    app.exec_()