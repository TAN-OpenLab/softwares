from MainWidget import MainWidget
from PyQt5.QtWidgets import QApplication
import sys
def main():
    app = QApplication(sys.argv)
    mainWidget = MainWidget()
    mainWidget.show()
    exit(app.exec_())
if __name__ == '__main__':
    main()
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPen, QColor, QSize
from PyQt5.QtCore import Qt
class PaintBoard(QWidget):
    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()
    def __InitView(self):
        self.setFixedSize(self.__size)
    def __InitData(self):
        # self.__size = QSize(480,460)
        # self.__size = QSize(280, 280)
        self.__size = QSize(600, 600)
        self.__board = QPixmap(self.__size)  # 新建QPixmap作为画板，宽350px,高250px
        self.__board.fill(Qt.black)  # 用白色填充画板
        self.__IsEmpty = True  # 默认为空画板
        self.EraserMode = False  # 默认为禁用橡皮擦模式
        self.__lastPos = QPoint(0, 0)
        self.__currentPos = QPoint(0, 0)
        self.__painter = QPainter()
        self.__painter.setRenderHints(QPainter.Antialiasing, True)
        self.__thickness = 60  # 默认画笔粗细为20px
        self.__penColor = QColor("white")  # 设置默认画笔颜色为黑色
        self.__colorList = QColor.colorNames()  # 获取颜色列表
    def Clear(self):
        # 清空画板
        self.__board.fill(Qt.black)
        self.update()
        self.__IsEmpty = True
    def ChangePenColor(self, color="white"):
        # 改变画笔颜色
        self.__penColor = QColor(color)
    def ChangePenThickness(self, thickness=20):
        # 改变画笔粗细
        self.__thickness = thickness
    def IsEmpty(self):
        # 返回画板是否为空
        return self.__IsEmpty
    def GetContentAsQImage(self):
        # 获取画板内容（返回QImage）
        image = self.__board.toImage()
        return image
    def paintEvent(self, paintEvent):
        self.__painter.begin(self)
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()
    def mousePressEvent(self, mouseEvent):
        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos
    def mouseMoveEvent(self, mouseEvent):
        self.__currentPos = mouseEvent.pos()
        self.__painter.begin(self.__board)
        if self.EraserMode == False:
            # 非橡皮擦模式
            self.__painter.setPen(QPen(self.__penColor, self.__thickness, Qt.SolidLine, Qt.RoundCap))  # 设置画笔颜色，粗细
        else:
            # 橡皮擦模式下画笔为纯黒色，粗细为20
            self.__painter.setPen(QPen(Qt.black, 60, Qt.SolidLine, Qt.RoundCap))
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        # self.__painter.drawArc()
        self.__painter.end()
        self.__lastPos = self.__currentPos
        self.update()  # 更新显示
    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False  # 画板不再为空
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
        self.setWindowTitle("Handwritten Digit Recognition Based on HOG-ELM")
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
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
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
        image = Image.fromarray(np.reshape(x_test[random_id], (28, 28)) * 255).convert('L')
        new_image = image.resize((44, 44))
        # new_image.show()
        new_image.save('tpic.png')
        pixmap = QPixmap('tpic.png')
        self.__label_timage.setPixmap(pixmap)
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
        y_hat = np.dot(self.w_out,
                       self.sigmoid(np.dot(self.w_in, np.reshape(fd, (length, 1))) + self.biase.transpose()))
        res_id = np.argmax(y_hat)
        y_hat[y_hat < 0] = 0
        prapability = y_hat[res_id] / np.sum(y_hat)
        myres = "{0}({1}%)".format(res_id, int(prapability * 100))
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
import pickle
import gzip
import numpy as np
def load_data():
    f = gzip.open('I:/wangpengfei-D/DeepLearning_Library/neural-networks-and-deep-learning-master/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    # print (training_data)
    # print (validation_data)
    # print (test_data)
    f.close()
    return (training_data, validation_data, test_data)
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    #training_data
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    #validation_data
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_result = [vectorized_result(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, training_results)
    #test_data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)
    return images, labels
def fashionmnist_loader():
    x_train, y_train = load_mnist('I:\wangpengfei-D\DeepLearning_Library\\fashionMnist')
    x_test, y_test = load_mnist('I:\wangpengfei-D\DeepLearning_Library\\fashionMnist', 't10k')
    y_tr = np.zeros((len(x_train), len(x_train[0])))
    for i in range(len(x_train)):
        y_tr[i][int(y_train[i])] = 1.0
    return x_train/255.0, y_tr, x_test/255.0, y_test
import numpy as np
import pickle
from sklearn.decomposition import PCA
def smallnorb_loader():
    f_x_train = open('I:\wangpengfei-D\pycodeLIB\\network\paperEx\smallnorb_x_train_24300x2048.pkl', 'rb')
    x_train = pickle.load(f_x_train, encoding='bytes')
    f_y_train = open('I:\wangpengfei-D\pycodeLIB\\network\paperEx\smallnorb_y_train.pkl', 'rb')
    y_train = pickle.load(f_y_train, encoding='bytes')

    f_x_test = open('I:\wangpengfei-D\pycodeLIB\\network\paperEx\smallnorb_x_test_24300x2048.pkl', 'rb')
    x_test = pickle.load(f_x_test, encoding='bytes')
    f_y_test = open('I:\wangpengfei-D\pycodeLIB\\network\paperEx\smallnorb_y_test.pkl', 'rb')
    y_test = pickle.load(f_y_test, encoding='bytes')
    return x_train, y_train, x_test, y_test
import numpy as np
import mnist_loader
import fashionmnist_loader
import smallNORBpkl_loader
import matplotlib.pyplot as plt
def get_Dataset(name='mnist'):
    if name == 'mnist':
        t, v, tt = mnist_loader.load_data_wrapper()
        validation_data = list(v)
        training_data = list(t) + validation_data
        testing_data = list(tt)
        # print (np.array(validation_data).shape)
        # print (np.reshape(validation_data[0][0], (784, 1)))
        # print(validation_data[0][1])
        # plt.subplot(1,1,1)
        # plt.imshow(np.reshape(validation_data[0][0], (28, 28)), 'gray')
        # plt.show()
        len_t = len(training_data)
        len_tdi = len(training_data[0][0])
        len_tl = len(training_data[0][1])
        x_train = np.zeros((len_t, len_tdi))
        y_train = np.zeros((len_t, len_tl))
        for i in range(len_t):
            x_train[i] = np.array(training_data[i][0]).transpose()
            y_train[i] = np.array(training_data[i][1]).transpose()

        len_tt = len(testing_data)
        x_test = np.zeros((len_tt, len_tdi))
        y_test = np.zeros(len_tt)
        for i in range(len_tt):
            x_test[i] = np.array(testing_data[i][0]).transpose()
            y_test[i] = testing_data[i][1]
        return x_train, y_train, x_test, y_test
    elif name == 'fashion':
        return fashionmnist_loader.fashionmnist_loader()
    elif name == 'smallnorb':
        x_train, y_tr, x_test, y_test = smallNORBpkl_loader.smallnorb_loader()
        length = len(y_tr)
        y_train = np.zeros((length, 5))
        for i in range(length):
            y_train[i][int(y_tr[i])] = 1.0
        return x_train, y_train, x_test, y_test
    else:
        pass
import dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt
dataset = dataset.SmallNORBDataset(dataset_root='I:\wangpengfei-D\DeepLearning_Library\smallNORB')
print(type(dataset))
########################### x_train ############################
examples_train_dat = dataset._parse_NORB_dat_file('I:\wangpengfei-D\DeepLearning_Library\smallNORB\smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
print (type(examples_train_dat))
print (len(examples_train_dat[0]))
print (np.array(examples_train_dat[0]).shape)
x_train = examples_train_dat.reshape(48600, 96*96)
print (np.array(x_train).shape)
print (x_train)
output = open('smallnorb_x_train.pkl', 'wb')
pickle.dump(x_train, output)
########################### x_test ############################
examples_test_dat = dataset._parse_NORB_dat_file('I:\wangpengfei-D\DeepLearning_Library\smallNORB\smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
print (type(examples_test_dat))
print (len(examples_test_dat[0]))
print (np.array(examples_test_dat[0]).shape)
x_test = examples_test_dat.reshape(48600, 96*96)
print (np.array(x_test).shape)
print (x_test)
output = open('smallnorb_x_test.pkl', 'wb')
pickle.dump(x_test, output)
########################### y_train ############################
examples_train_cat = dataset._parse_NORB_cat_file('I:\wangpengfei-D\DeepLearning_Library\smallNORB\smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat')
print (type(examples_train_cat))
# print (len(examples_train_cat[0]))
# print (np.array(examples_train_cat[0]).shape)
y_train = examples_train_cat
print (np.array(y_train).shape)
print (y_train)
output = open('smallnorb_y_train.pkl', 'wb')
pickle.dump(y_train, output)
########################### y_test ############################
examples_test_cat = dataset._parse_NORB_cat_file('I:\wangpengfei-D\DeepLearning_Library\smallNORB\smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat')
print (type(examples_test_cat))
y_test = examples_test_cat
print (np.array(y_test).shape)
print (y_test)
output = open('smallnorb_y_test.pkl', 'wb')
pickle.dump(y_test, output)
########################### recover some samples to pictures ############################
for i in range(200):
    plt.subplot(20,10,i+1)
    plt.imshow(x_test[i].reshape(96, 96), 'gray')
    plt.axis('off')
    plt.pause(0.0001)
plt.savefig('I:\wangpengfei-D\pycodeLIB\\network\pic\\smallnorb_test_25.png')
plt.show()
import json
import numpy as np
import time
import get_Dataset
from skimage.feature import hog
from scipy.sparse import identity
from scipy.stats.mstats import zscore
import pickle
from scipy.linalg import orth
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    # return
class ELM(object):
    def __init__(self, sizes, len_training_data):
        self.lmax = 0
        self.num_layers = len(sizes)
        self.num_training_data = len_training_data
        self.sizes = sizes
        self.biase = np.random.uniform(-0.2, 0.2, (1, self.sizes[1]))
        self.biase = orth(self.biase)
        self.biases = np.zeros((len_training_data, self.sizes[1]))
        self.W_in = np.random.uniform(-1.0, 1.0, (self.sizes[1], self.sizes[0]))
        if self.sizes[1] > self.sizes[0]:
            self.W_in = orth(self.W_in)
            # print (self.W_in.shape)
        else:
            self.W_in = orth(self.W_in.T).T
            # print (self.W_in.shape)
        # self.W_in = orth(self.W_in)
        self.W_out = np.zeros((self.sizes[2], self.sizes[1]))
        self.H = np.zeros((self.num_training_data, self.sizes[1]))
        for i in range(len_training_data):
            self.biases[i] = self.biase
    def build_H_update_beta(self, train_data, train_label, C):
        # self.H = sigmoid(np.dot(train_data, self.W_in.transpose()) + self.biases)
        self.H = np.dot(train_data, self.W_in.transpose()) + self.biases
        self.lmax = 1.0 / np.max(np.max(self.H, axis=1))
        self.H = sigmoid(self.H*self.lmax)
        T = train_label
        if self.num_training_data > self.sizes[1]:
            E = np.eye(self.sizes[1])
            first = E * C + np.dot(self.H.transpose(), self.H)
            first_plus = np.linalg.inv(first)
            second = np.dot(self.H.transpose(), T)
            self.W_out = np.dot(first_plus, second).transpose()
            # self.W_out = np.linalg.solve(np.dot(self.H.T, self.H) + np.eye(self.H.shape[1])*C, np.dot(self.H.T, T)).T
        else:
            E = identity(self.num_training_data).toarray()
            first = E * C + np.dot(self.H, self.H.transpose())
            first_plus = np.linalg.inv(first)
            second = np.dot(first_plus, T)
            self.W_out = np.dot(self.H.transpose(), second).transpose()
    def save(self, filename):
        data = {"W_in": self.W_in.tolist(),
                "W_out": self.W_out.tolist(),
                "biase": self.biase.tolist(),
                "lmax": [self.lmax]
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    def evaluation(self, x_test, y_test):
        sum = 0
        len_test = len(x_test)
        length = len(x_test[0])
        for i in range(len_test):
            # y_hat = np.dot(self.W_out, sigmoid(np.dot(self.W_in, np.reshape((x_test[i]), (length, 1))) + self.biase.transpose()))
            y_hat = np.dot(self.W_out, sigmoid((np.dot(self.W_in, np.reshape((x_test[i]), (length, 1))) + self.biase.transpose())*self.lmax))
            if np.argmax(y_hat) == y_test[i]:
                sum += 1
        return sum*100.0/len_test
def hog_ELM(direction_num, cells, blocks, hidden_nodes, x_train, y_train, x_test, y_test):
    len_t = len(x_train)
    len_tt = len(x_test)
    #pick features
    start1 = time.clock()
    hog_train_data = []
    for i in range(len_t):
        fd = hog(x_train[i].reshape((28, 28)),
                 orientations=direction_num,
                 pixels_per_cell=(cells, cells),
                 cells_per_block=(blocks, blocks),
                 block_norm='L1-sqrt',
                 transform_sqrt=False,
                 feature_vector=True,
                 visualise=False
                 )
        hog_train_data.append(fd)
    hog_test_data = []
    for i in range(len_tt):
        fdd = hog(x_test[i].reshape((28, 28)),
                 orientations=direction_num,
                 pixels_per_cell=(cells, cells),
                 cells_per_block=(blocks, blocks),
                 block_norm='L1-sqrt',
                 transform_sqrt=False,
                 feature_vector=True,
                 visualise=False
                  )
        hog_test_data.append(fdd)
    end1 = time.clock()
    hog_test_datalie = []
    length = len(hog_test_data[0])
    for i in range(len_tt):
        hog_test_datalie.append(np.reshape(hog_test_data[i], (length, 1)))
    print (length)
    start = time.clock()
    elm = ELM([length, hidden_nodes, 10], len_t)
    elm.build_H_update_beta(hog_train_data, y_train, 2**(-30))
    end = time.clock()
    return (elm.evaluation(hog_test_datalie, y_test)), (end - start)
xo_train, y_train, xo_test, y_test=get_Dataset.get_Dataset(name='mnist')
f = open('mnist_train_hog.pkl', 'rb')
x_train = pickle.load(f, encoding='bytes')
f1 = open('mnist_test_hog.pkl', 'rb')
x_test = pickle.load(f1, encoding='bytes')
print (hog_ELM(5,4,5,5000,xo_train,y_train,xo_test,y_test))