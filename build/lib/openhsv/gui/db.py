from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, \
    QGridLayout, QLineEdit, QTableWidget, QDateEdit, QTableWidgetItem, \
    QHeaderView, QTreeView 
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
import pyqtgraph as pg 
from glob import glob
import imageio as io
import numpy as np
import json


class Table(QWidget):
    def __init__(self, folder):
        """Table widget for showing patient data.

        :param folder: folder that contains patient data
        :type folder: str
        """
        super().__init__()
        self.l = QGridLayout(self)
        self.setMinimumWidth(700)

        # Search options
        self.surname = QLineEdit()
        self.surname.setPlaceholderText("Last name, e.g. Smith")
        self.surname.textChanged.connect(self.search)
        self.firstname = QLineEdit()
        self.firstname.setPlaceholderText("First name")
        self.firstname.textChanged.connect(self.search)
        self.birthdate = QDateEdit()
        self.birthdate.dateChanged.connect(self.search)
        self.search = QPushButton("Search")

        self.l.addWidget(self.surname, 0, 0)
        self.l.addWidget(self.firstname, 0, 1)
        self.l.addWidget(self.birthdate, 0, 2)
        self.l.addWidget(self.search, 0, 3)

        self.folder = folder
        self.im = None

        # Create empty table
        self.t = QTableWidget()
        h = self.t.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Stretch)
        
        # Look for patient data in given folder
        self.fns = glob(folder+"\\**\\*.meta", recursive=True)
        self.patients = []
        self.opened = ""

        # Iterate over metadata files
        for ii, fn in enumerate(self.fns):
            meta = json.load(open(fn))

            meta['Patient']['date'] = meta['Date']

            # Store patient metadata
            self.patients.append(meta['Patient'])

            rc = self.t.rowCount()
            self.t.setRowCount(rc+1)

            # Set column when first file is loaded
            if ii == 0:
                self.t.setColumnCount(len(self.patients[0].keys()))
                self.t.setHorizontalHeaderLabels(self.patients[0].keys())

            # Show metadata in a single row
            for i, (k, v) in enumerate(self.patients[-1].items()):
                item = QTableWidgetItem(v)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.t.setItem(rc, i, item)

        # connect 
        self.t.doubleClicked.connect(self.do)

        self.l.addWidget(self.t, 1, 0, 4, 10)

    def search(self):
        """Search for entries in database
        """
        # Get search terms
        surname = self.surname.text()
        firstname = self.firstname.text()
        birthdate = self.birthdate.text()

        # Iterate over rows
        # Only show rows that contain relevant data
        # As entered/selected in search
        for i in range(self.t.rowCount()):
            self.t.showRow(i)

            if surname:
                if surname not in self.t.item(i, 0).text():
                    self.t.hideRow(i)

            if firstname:
                if firstname not in self.t.item(i, 1).text():
                    self.t.hideRow(i)

            if birthdate != QDateEdit().text():
                if birthdate != self.t.item(i, 3).text():
                    self.t.hideRow(i)

    def do(self, e):
        """open selected patient

        :param e: event information
        :type e: QModelIndex
        """
        # Get clicked row
        row = e.row()

        # Get respective file name and look for video
        fn = self.fns[row]
        vid_fn = fn.replace(".meta", ".mp4")

        # If video is already opened, do nothing
        if vid_fn == self.opened and type(self.im) != None:
            if self.im.isVisible():
                return
        
        # Otherwise, read video
        vid = io.mimread(vid_fn,
            memtest=False)

        # Show video
        self.im = pg.image(np.asarray(vid, np.uint8).transpose(0,2,1,3),
            title=vid_fn)

        meta = json.load(open(fn))
        self.d = DictViewer(meta)
        self.d.show()

        self.opened = vid_fn


class DictViewer(QTreeView):
    def __init__(self, dictionary):
        """Shows a dictionary as a tree view in a separate window.

        :param dictionary: the JSON-like dictionary that should be shown
        :type dictionary: dict
        """
        super().__init__()
        self.d = dictionary

        # Use a default tree structure with two columns
        model = QStandardItemModel()
        self.setModel(model)
        model.setColumnCount(2)
        model.setHorizontalHeaderLabels(["key", "value"])

        # For each main node
        for k, v in self.d.items():
            parent = QStandardItem(k)

            # If sub-nodes are available
            if type(v) is dict:                
                for k1, v1 in v.items():
                    parent1 = QStandardItem(k1)
                    c1 = QStandardItem(str(v1))

                    parent.appendRow([parent1, c1])

            else:
                cs = [QStandardItem(v)]
                parent.appendRow(cs)    

            model.appendRow(parent)


class DB(QMainWindow):
    def __init__(self, folder):
        """Main database window

        :param folder: folder with patient data 
        :type folder: str
        """
        super().__init__()
        self.l = QGridLayout(self)
        self.t = Table(folder)
        self.folder = folder

        self.s = self.statusBar()
        self.s.showMessage(f"Data location: {self.folder}")

        self.setCentralWidget(
            self.t
        )

        self.setWindowTitle("Patient Database Viewer")



if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication([])

    db = DB("C:/openhsv")
    db.show()

    sys.exit(app.exec_())