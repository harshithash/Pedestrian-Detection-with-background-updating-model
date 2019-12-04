from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(724, 541)
        MainWindow.setFocusPolicy(QtCore.Qt.ClickFocus)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 0, 671, 501))
        self.frame.setStyleSheet(_fromUtf8("background-color: rgb(77, 154, 231);\n"
""))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.input_dir = QtGui.QPushButton(self.frame)
        self.input_dir.setGeometry(QtCore.QRect(30, 30, 231, 31))
        self.input_dir.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.input_dir.setStyleSheet(_fromUtf8("background-color: rgb(211,211,211);\n"
"selection-color: rgb(204, 204, 204);\n"
"font: 63 12pt \"Ubuntu\";\n"
"\n"
"\n"
"border-color: rgb(132, 132, 132);\n"
""))
        self.input_dir.setObjectName(_fromUtf8("input_dir"))
        self.output_dir = QtGui.QPushButton(self.frame)
        self.output_dir.setGeometry(QtCore.QRect(30, 80, 231, 31))
        self.output_dir.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.output_dir.setStyleSheet(_fromUtf8("background-color: rgb(211,211,211);\n"
"selection-color: rgb(204, 204, 204);\n"
"font: 63 12pt \"Ubuntu\";\n"
"\n"
"\n"
"border-color: rgb(132, 132, 132);\n"
""))
        self.output_dir.setObjectName(_fromUtf8("output_dir"))
        self.bg_img = QtGui.QPushButton(self.frame)
        self.bg_img.setGeometry(QtCore.QRect(30, 130, 231, 31))
        self.bg_img.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.bg_img.setStyleSheet(_fromUtf8("background-color: rgb(211,211,211);\n"
"selection-color: rgb(204, 204, 204);\n"
"font: 63 12pt \"Ubuntu\";\n"
"\n"
"\n"
"border-color: rgb(132, 132, 132);\n"
""))
        self.bg_img.setObjectName(_fromUtf8("bg_img"))
        self.inp_dir_view = QtGui.QLineEdit(self.frame)
        self.inp_dir_view.setGeometry(QtCore.QRect(290, 30, 351, 27))
        self.inp_dir_view.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);"))
        self.inp_dir_view.setObjectName(_fromUtf8("inp_dir_view"))
        self.bg_img_view = QtGui.QLineEdit(self.frame)
        self.bg_img_view.setGeometry(QtCore.QRect(290, 130, 351, 27))
        self.bg_img_view.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);"))
        self.bg_img_view.setObjectName(_fromUtf8("bg_img_view"))
        self.out_dir_view = QtGui.QLineEdit(self.frame)
        self.out_dir_view.setGeometry(QtCore.QRect(290, 80, 351, 27))
        self.out_dir_view.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);"))
        self.out_dir_view.setObjectName(_fromUtf8("out_dir_view"))
        self.frame_2 = QtGui.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(30, 210, 611, 161))
        self.frame_2.setStyleSheet(_fromUtf8("background-color: rgb(211, 211, 211);"))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setLineWidth(3)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.label_2 = QtGui.QLabel(self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 191, 31))
        self.label_2.setStyleSheet(_fromUtf8("font: 12pt \"TakaoPGothic\";\n"
"background-color: rgb(184, 184, 184);"))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.frame_2)
        self.label_3.setGeometry(QtCore.QRect(310, 20, 211, 31))
        self.label_3.setStyleSheet(_fromUtf8("font: 12pt \"TakaoPGothic\";\n"
"background-color: rgb(184, 184, 184);"))
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.frame_2)
        self.label_4.setGeometry(QtCore.QRect(310, 60, 211, 41))
        self.label_4.setStyleSheet(_fromUtf8("font: 12pt \"TakaoPGothic\";\n"
"background-color: rgb(184, 184, 184);"))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.frame_2)
        self.label_5.setGeometry(QtCore.QRect(10, 60, 191, 41))
        self.label_5.setStyleSheet(_fromUtf8("font: 12pt \"TakaoPGothic\";\n"
"background-color: rgb(184, 184, 184);"))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.n_centers_inp = QtGui.QLineEdit(self.frame_2)
        self.n_centers_inp.setGeometry(QtCore.QRect(210, 20, 71, 27))
        self.n_centers_inp.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);"))
        self.n_centers_inp.setObjectName(_fromUtf8("n_centers_inp"))
        
        self.key_thresh_inp = QtGui.QLineEdit(self.frame_2)
        self.key_thresh_inp.setGeometry(QtCore.QRect(530, 20, 71, 27))
        self.key_thresh_inp.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);"))
        self.key_thresh_inp.setObjectName(_fromUtf8("key_thresh_inp"))
        
        self.bg_color_frame_inp = QtGui.QLineEdit(self.frame_2)
        self.bg_color_frame_inp.setGeometry(QtCore.QRect(530, 70, 71, 27))
        self.bg_color_frame_inp.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);"))
        self.bg_color_frame_inp.setObjectName(_fromUtf8("bg_color_frame_inp"))
        
        self.bg_cluster_thresh_inp = QtGui.QLineEdit(self.frame_2)
        self.bg_cluster_thresh_inp.setGeometry(QtCore.QRect(210, 70, 71, 27))
        self.bg_cluster_thresh_inp.setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);"))
        self.bg_cluster_thresh_inp.setObjectName(_fromUtf8("bg_cluster_thresh_inp"))
        
        self.set_btn = QtGui.QPushButton(self.frame_2)
        self.set_btn.setGeometry(QtCore.QRect(490, 120, 98, 27))
        self.set_btn.setStyleSheet(_fromUtf8("background-color: rgb(117, 235, 174);"))
        self.set_btn.setObjectName(_fromUtf8("set_btn"))
        
        self.label = QtGui.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(40, 190, 141, 31))
        self.label.setStyleSheet(_fromUtf8("background-color: rgb(184, 184, 184);\n"
"border-color: rgb(136, 136, 136);\n"
"font: 14pt \"TakaoPGothic\";"))
        self.label.setFrameShadow(QtGui.QFrame.Raised)
        self.label.setLineWidth(2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMargin(1)
        self.label.setIndent(-1)
        self.label.setObjectName(_fromUtf8("label"))
        
        self.run_btn = QtGui.QPushButton(self.frame)
        self.run_btn.setGeometry(QtCore.QRect(240, 400, 221, 41))
        self.run_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.run_btn.setStyleSheet(_fromUtf8("background-color: rgb(96, 193, 143);\n"
"\n"
"selection-color: rgb(204, 204, 204);\n"
"font: 75 14pt \"TakaoPGothic\";\n"
"\n"
"\n"
"border-color: rgb(132, 132, 132);\n"
""))
        self.run_btn.setObjectName(_fromUtf8("run_btn"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 724, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.input_dir.setText(_translate("MainWindow", "Browse Input Directory", None))
        self.output_dir.setText(_translate("MainWindow", "Set Output Directory Path ", None))
        self.bg_img.setText(_translate("MainWindow", "Browse Background Image", None))
        self.label_2.setToolTip(_translate("MainWindow", "<html><head/><body><p>Optimal number of colors in which the background can be clustered.</p></body></html>", None))
        self.label_2.setText(_translate("MainWindow", "No of Clusters", None))
        self.label_3.setToolTip(_translate("MainWindow", "<html><head/><body><p>Determines the minimum absolute difference for selection of the key frames.</p></body></html>", None))
        self.label_3.setText(_translate("MainWindow", "Key Frame Threshold", None))
        self.label_4.setToolTip(_translate("MainWindow", "<html><head/><body><p>Number of pixels that need to be updated for reclustering of background.</p></body></html>", None))
        self.label_4.setText(_translate("MainWindow", "BG Reclustering Threshold", None))
        self.label_5.setToolTip(_translate("MainWindow", "<html><head/><body><p>Minimum Number of frames after which the new pixel is marked for updation.</p></body></html>", None))
        self.label_5.setText(_translate("MainWindow", "BG Pixel Color Update Threshold", None))
        self.set_btn.setText(_translate("MainWindow", "Set", None))
        self.label.setText(_translate("MainWindow", "Parameters", None))
        self.run_btn.setText(_translate("MainWindow", "RUN", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))

