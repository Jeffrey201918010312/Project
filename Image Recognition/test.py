from tools.datatools import load_ds
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import AUC, Recall, Precision
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
from predict import RecTool
from train import read_datas
import tensorflow as tf


def main(data_dir="datasets/test"):
    labels, test_ds = read_datas(data_dir)
    labels = [lb.split("-")[0] for lb in labels]
    test_ds = load_ds(*test_ds)
    y_test = np.concatenate([y for _, y in test_ds], axis=0)
    y_test = np.array(y_test)
    print(f"{len(y_test)}")
    y_test = np.argmax(y_test, axis=1)
    test_model = RecTool()
    y_pred = test_model.model.predict(test_ds)
    opt = tf.keras.optimizers.SGD(learning_rate=0.001, decay=0.0001, momentum=0.9)
    test_model.model.compile(optimizer=opt,
                             metrics=["accuracy",  
                                      AUC(),  
                                      Recall(),  
                                      Precision()  
                                      ],
                             loss=[CategoricalCrossentropy()])  
    res = test_model.model.evaluate(test_ds)
    for sc, re in zip(["Losses", "Accuracy", "AUC", "Recall", "Precision"], res):
        print(sc, re)
    plt.figure()
    cf = confusion_matrix(y_test, tf.argmax(y_pred, axis=1), labels=[i for i in range(len(labels))])
    ax = sns.heatmap(cf, annot=True, xticklabels=labels, yticklabels=labels)
    ax.tick_params(labelleft=True, labeltop=True, labelbottom=False)
    plt.title("Confusion matrix")
    plt.savefig("Confusion matrix")
    plt.close()
    plt.show()


if __name__ == '__main__':
    main()



