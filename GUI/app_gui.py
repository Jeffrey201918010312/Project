import os
import sys
import cv2
import tensorflow as tf
from tools.utils import imread
import numpy as np
from app import Ui_MainWindow
from PyQt5 import Qt
import PyQt5.QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow
from nets.networks import MyVgg16, CnnModel
CUDA = False


class MyWindow(QMainWindow, Ui_MainWindow):

    if not CUDA:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.q_img = None
        self.image = None
        self.model_path = os.path.join(os.getcwd(), "save/best_epoch_weights.h5")
        self.input_shape = 224
        self.model_name = "resnet"
        self.model = None
        self.labels = None
        self.buttons_init()
        self.init_model()
        self.setWindowTitle("plane classifier")
        self.label.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.QColor(205, 205, 215))  #
        self.label.setPalette(palette)

    def buttons_init(self):
        self.openFile.clicked.connect(self.openFileFunc)
        self.pushButton.clicked.connect(self.recFunc)

    def recFunc(self):
        if self.image is not None:
            try:
                image = cv2.resize(self.image, (self.input_shape, self.input_shape))
                image = np.array(image, dtype=np.float32) / 255. - 0.5
                image = tf.reshape(image, [-1, self.input_shape, self.input_shape, 3])
            except:
                self.warning(info="loading data error")
                # print(image.shape, type(image), image.dtype)
                return
            try:
                # res = self.model.predict(image)
                # res = np.array([[1,0,0,0]])
                res = self.model(image)
            except:
                self.setData(None)
                self.warning(info="something went wrong so it's failed to recognition")
                return
            # print(res)
            idx = np.argmax(res)
            # print(idx)
            if res[0, idx] > 0.4:
                res = self.labels[idx % len(self.labels)]
            else:
                res = "None"
            self.resDir.setText(res)


    def openFileFunc(self):
        path_list = QFileDialog.getOpenFileName(self, "select file", "./", "Images (*.jpg;*.png;*.jpeg;*.*)")
        img_path = path_list[0]
        self.pathDir.setText(img_path)
        self.resDir.setText("")
        self.setData(None)
        if img_path == "":
            self.warning("warning", "didn't select file")
            return
        if os.path.isfile(img_path):
            self.setData(img_path)
        else:
            self.warning(info="can not find the file!")

    def resizeEvent(self, a0: QtGui.QResizeEvent):
        self.showQimg()

    def image2qimage(self, image, input_type="RGB"):
        spl = len(image.shape)
        # print(spl)
        if spl != 2 and spl != 3:
            return QImage()
        if spl == 2:
            self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if input_type == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_img = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
        return q_img

    def showQimg(self):
        if self.q_img is None:
            pix = QPixmap()
            self.label.setPixmap(pix)
            return
        width = self.label.width()
        height = self.label.height()
        if self.q_img.width() != 0 and self.q_img.height() != 0:
            pix = QPixmap(self.q_img).scaled(width, height)
            self.label.setPixmap(pix)
        print("creating window...")

    def warning(self, title='waring', info="failed to load image"):
        PyQt5.QtWidgets.QMessageBox.warning(self, title, info)

    def setData(self, param):
        if param is None:
            self.pathDir.setText("")
            self.resDir.setText("")
            self.q_img = None
            self.image = None
        else:
            try:
                self.pathDir.setText(param)
                self.image = imread(param)
                if self.image is None:
                    self.warning(info="please select the right file")
                    self.setData(None)
                self.q_img = self.image2qimage(self.image, input_type="RGB")
                if self.q_img is None:
                    self.warning()
                    self.setData(None)
            except:
                self.setData(None)
        self.showQimg()

    def init_model(self):
        f = open(os.path.join(os.getcwd(), "save/classes.txt"), mode="r", encoding="utf-8")
        self.labels = f.readline().split()
        f.close()
        self.model = CnnModel(len(self.labels), self.input_shape, self.model_name)
        self.model.build(input_shape=(None, self.input_shape, self.input_shape, 3))
        self.model.load_weights(self.model_path)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.resize(960, 640)
    myWindow.show()
    sys.exit(app.exec_())
