import numpy as np
import PyQt5
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog
from PyQt5 import QtGui
from PaintBoard import PaintBoard
from PyQt5.QtCore import *
from PIL import Image
import json
from skimage.feature import hog

import pickle
import random



class MainWidget(QWidget):


    def __init__(self, Parent=None):

        super().__init__(Parent)
        
        self.__InitData()
        self.__InitView()
        f = open("params15000.json", "r")
        data = json.load(f)
        f.close()
        self.biase = np.array(data["biase"])
        self.w_in = np.array(data["w_in"])
        self.w_out = np.array(data["w_out"])
    
    def __InitData(self):

        self.__paintBoard = PaintBoard(self)
        self.__colorList = QColor.colorNames()
        
    def __InitView(self):

        # self.setFixedSize(640,480)
        # self.setFixedSize(480,410)
        self.setFixedSize(800, 618)
        self.setWindowTitle("基于ELM的快速手写识别软件NeuHandWriting (NeuHWR1.0)")
        self.setWindowIcon(QIcon('ICON.ico'))
        
        
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)

        
        main_layout.addWidget(self.__paintBoard)

        self.font = QtGui.QFont()
        self.font.setFamily("Consolas")
        self.font.setPointSize(15)

        
        sub_layout = QVBoxLayout()
        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Clear = QPushButton("Clear Board")
        self.__btn_Clear.setParent(self)
        self.__btn_Clear.setFixedSize(146, 70)
        self.__btn_Clear.setFont(self.font)
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)
        
        self.__btn_Quit = QPushButton("Exit")
        self.__btn_Quit.setParent(self)
        self.__btn_Quit.setFixedSize(146, 70)
        self.__btn_Quit.setFont(self.font)
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)
        
        self.__btn_Save = QPushButton("Save Board")
        self.__btn_Save.setParent(self)
        self.__btn_Save.setFixedSize(146, 70)
        self.__btn_Save.setFont(self.font)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)
        
        self.__cbtn_Eraser = QCheckBox("Eraser")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.setFont(self.font)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)
        


        self.__label_result = QLabel(self)
        self.__label_result.setAlignment(Qt.AlignCenter)
        self.__label_result.setFixedWidth(146)
        self.__label_result.setFixedHeight(80)
        self.__label_result.setFrameStyle(PyQt5.QtWidgets.QFrame.Box)
        self.__label_result.setFrameShadow(PyQt5.QtWidgets.QFrame.Raised)
        self.__label_result.setLineWidth(6)
        self.__label_result.setMidLineWidth(5)
        self.__label_result.setStyleSheet('background-color: rgb(255,123,100)')
        sub_layout.addWidget(self.__label_result)


        self.__btn_Recognize = QPushButton("Recognition")
        self.__btn_Recognize.setParent(self)
        self.__btn_Recognize.setFixedSize(146, 70)
        self.__btn_Recognize.setFont(self.font)
        self.__btn_Recognize.clicked.connect(self.on_btn_Recognize_clicked)
        sub_layout.addWidget(self.__btn_Recognize)

        self.__label_timage = QLabel(self)
        self.__label_timage.setAlignment(Qt.AlignCenter)
        self.__label_timage.setFixedWidth(146)
        self.__label_timage.setFixedHeight(80)
        self.__label_timage.setFrameStyle(PyQt5.QtWidgets.QFrame.Box)
        self.__label_timage.setFrameShadow(PyQt5.QtWidgets.QFrame.Raised)
        self.__label_timage.setLineWidth(6)
        self.__label_timage.setMidLineWidth(5)
        self.__label_timage.setStyleSheet('background-color: rgb(125,143,50)')
        sub_layout.addWidget(self.__label_timage)





        self.__btn_Random = QPushButton("RandomDigit \nFrom TestSet")
        self.__btn_Random.setParent(self)
        self.__btn_Random.setFixedSize(146, 70)
        self.font.setPointSize(13)
        self.__btn_Random.setFont(self.font)
        self.__btn_Random.clicked.connect(self.on_btn_RandomDigit_Clicked)
        sub_layout.addWidget(self.__btn_Random)

        main_layout.addLayout(sub_layout)

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList: 
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70,20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix),None)
            comboBox.setIconSize(QSize(70,20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)
        
    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)
    
    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])
        
    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True
        else:
            self.__paintBoard.EraserMode = False

    def on_btn_RandomDigit_Clicked(self):
        f = open('x_test_10000.pkl', 'rb')
        x_test = pickle.load(f, encoding='bytes')
        f.close()
        ff = open('y_test_10000.pkl', 'rb')
        y_test = pickle.load(ff, encoding='bytes')
        ff.close()
        random_id = random.randint(0, 9999)
        image = Image.fromarray(np.reshape(x_test[random_id], (28, 28))*255).convert('L')
        new_image = image.resize((44, 44))
        # new_image.show()
        new_image.save('tpic.png')
        pixmap = QPixmap('tpic.png')
        self.__label_timage.setPixmap(pixmap)
        print("begin")
        res = self.judge(np.reshape(x_test[random_id], (28, 28)))
        myres = "{0}|{1}".format(res, int(y_test[random_id]))

        # show my result
        font = QtGui.QFont()
        font.setFamily('Consolas')
        font.setBold(True)
        font.setPointSize(18)
        font.setWeight(75)
        self.__label_result.setFont(font)
        self.__label_result.setText(myres)


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def judge(self, input_v):
        fd = hog(input_v,
                 orientations=5,
                 pixels_per_cell=(4, 4),
                 cells_per_block=(3, 3),
                 block_norm='L1-sqrt',
                 transform_sqrt=True,
                 feature_vector=True,
                 visualise=False
                 )
        length = len(fd)
        y_hat = np.dot(self.w_out, self.sigmoid(np.dot(self.w_in, np.reshape(fd, (length, 1))) + self.biase.transpose()))
        res_id = np.argmax(y_hat)
        y_hat[y_hat < 0] = 0
        prapability = y_hat[res_id] / np.sum(y_hat)
        myres = "{0}({1}%)".format(res_id, int(prapability * 100))
        print (myres)
        return myres

    def on_btn_Recognize_clicked(self):

        savePath = "tmp.png"
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath)

        t_image = Image.open(savePath).convert('L')
        newt_image = t_image.resize((28, 28))
        # newt_image.save("tmp1.png")
        np_image = np.array(newt_image) / 255.0
        # print (np_image.shape)
        # print (np_image)

        myres = self.judge(np_image)

        # show my result
        font = QtGui.QFont()
        font.setFamily('Consolas')
        font.setBold(True)
        font.setPointSize(20)
        font.setWeight(75)
        self.__label_result.setFont(font)
        self.__label_result.setText(myres)
        # print (myres)

    def Quit(self):
        self.close()