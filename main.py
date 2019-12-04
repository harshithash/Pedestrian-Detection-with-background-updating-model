#!/usr/bin/python2.5
import sys
import os
from PyQt4.Qt import *
from PyQt4 import QtGui,QtCore
from PyQt4.QtGui import QFileDialog, QWidget
from output import *
import algo

class MyPopup(QWidget):
    t="abc"
    def __init__(self,text):
    	self.t=text
        QWidget.__init__(self)
        
    def paintEvent(self, e):
        dc = QPainter(self)
        dc.drawText(QPoint(10,50),self.t)
        
    
class MyForm(QtGui.QMainWindow):
	def selectInpDir(self):
    		self.ui.inp_dir_view.setText(QFileDialog.getExistingDirectory(self, "Select Directory"))
    	
    	def selectOutDir(self):
    		self.ui.out_dir_view.setText(QFileDialog.getExistingDirectory(self, "Select Directory"))
    		
    	def selectBgImg(self):
    		self.ui.bg_img_view.setText(QFileDialog.getOpenFileName(self, 'Open file', 
   '/',"Image files (*.jpg *.png *.jpeg)"))
   	
   	def popit(self,text):
		print "Opening a new popup window...",text
		self.w = MyPopup(text)
		self.w.setGeometry(QRect(100, 100, 600, 100))
		self.w.show()
    	
    	def selectSetBtn(self):
    		bgc = self.ui.inp_dir_view.text()
    		b= str(bgc)
    		if b == "":
    			self.popit("Select the Input Directory!!!")
    			return
    		else:
    			self.inp_dir=b
    				
    		bgc = self.ui.out_dir_view.text()
    		b= str(bgc)
    		if b == "":
    			self.popit("Select the Output Directory!!!")
    			return
    		else:
    			self.out_dir=b
    				
    		bgc = self.ui.bg_img_view.text()
    		b= str(bgc)
    		if b == "":
    			self.popit("Select the Background Image!!!")
    			return
    		else:
    			self.bg_img=b
    			
    		bgc = self.ui.n_centers_inp.text()
    		(a,b)=bgc.split(" ")[0].toInt(); 
    		if b == True and a > 2 and a < 10:
    			self.n_centers=a
    		else:
    			self.popit("No of Clusters Error: Enter an integer between 2 and 10!!!")
    			return
    			
    		bgc = self.ui.key_thresh_inp.text()
    		(a,b)=bgc.split(" ")[0].toInt()
    		if b == True and a > 100:
    			a=a/100.0
    			self.key_thresh=a
    		else:
    			self.popit("Key frame threshold Error : Enter an integer above 100!!!")
    			return
    			
    		bgc = self.ui.bg_cluster_thresh_inp.text()
    		(a,b)=bgc.split(" ")[0].toInt()
    		if b == True and a > 5:
    			self.bg_cluster_thresh=a
    		else:
    			self.popit("BG Pixel Color Update Threshold Error : Enter an integer above 5!!!")
    			return
    		
    		bgc = self.ui.bg_color_frame_inp.text()
    		(a,b)=bgc.split(" ")[0].toInt()
    		if b == True and a > 5:
    			self.bg_color_frame=a
    		else:
    			self.popit("BG Reclustering Threshold Error : Enter an integer above 50!!!")
    			return
    			
    	def selectRunBtn(self):
		algo.main1(self.inp_dir,self.out_dir,self.bg_img,self.n_centers,self.key_thresh,self.bg_cluster_thresh,self.bg_color_frame)
    		
        def __init__(self, parent=None):
                #build parent user interface
                QtGui.QWidget.__init__(self, parent)
                self.ui = Ui_MainWindow()
                self.ui.setupUi(self)
                
                #connect buttons
                QtCore.QObject.connect(self.ui.input_dir, QtCore.SIGNAL('clicked()'), self.selectInpDir)
		QtCore.QObject.connect(self.ui.output_dir, QtCore.SIGNAL('clicked()'), self.selectOutDir)
		QtCore.QObject.connect(self.ui.bg_img, QtCore.SIGNAL('clicked()'), self.selectBgImg)
		QtCore.QObject.connect(self.ui.set_btn, QtCore.SIGNAL('clicked()'), self.selectSetBtn)
		QtCore.QObject.connect(self.ui.run_btn, QtCore.SIGNAL('clicked()'), self.selectRunBtn)
		
if __name__ == "__main__":
        #This function means this was run directly, not called from another python file.
        app = QtGui.QApplication(sys.argv)
        myapp = MyForm()
        myapp.show()
        sys.exit(app.exec_())
        

