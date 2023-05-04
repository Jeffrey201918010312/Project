import os
import numpy as np
import tensorflow as tf
from nets.networks import MyVgg16, CnnModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]

class RecTool():
    def __init__(self, model_name="resnet", cuda=True):
        if not cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  
        self.model_name = model_name
        self.model_path = "save/best_epoch_weights.h5"
        self.input_shape = 224
        self.init()  
    def __call__(self, img_path, **kwargs):
        img = None
        image = None
        try:
            print(f"Loading images from {img_path}")
            img = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(image, [224, 224]) / 255 - 0.5
            img = tf.reshape(img, [1, 224, 224, 3])
            print("Image loaded successfully, recognition in progress...")
        except:
            print("Image failed to load, check file or path...")
        if img is None:
            return
        try:
            res = self.model(img)
            idx = np.argmax(res)
            res = self.labels[idx]
            plt.figure()
            plt.imshow(np.array(image))
            plt.axis("off")
            plt.title(res)
            plt.show()

        except:
            print("Recognition failed, please check model loading")

    def init(self):
        f = open("save/classes.txt", mode="r", encoding="utf-8")
        self.labels = f.readline().split()
        f.close()
        self.model = CnnModel(len(self.labels), self.input_shape, self.model_name)  
        self.model.build(input_shape=(None, self.input_shape, self.input_shape, 3))
        self.model.load_weights(self.model_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  
    model = RecTool()
    while True:
        img_path = input("Input image path(example:xxx//xxx.jpg): ('quit')\n")
        if str(img_path) == "quit":
            print("quit")
            break
        model(img_path)
