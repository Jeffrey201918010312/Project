import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tools.utils import get_pics_list
from tools.datatools import load_ds
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import AUC, Recall, Precision
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
from nets.networks import MyVgg16, CnnModel

def read_datas(root_dir):
    
    data_list = []
    label_list = []
    folders = os.listdir(root_dir)
    for i, folder in enumerate(folders): 
        pic_list = get_pics_list(root_dir, folder)
        if len(pic_list) > 0:
            lb_list = [i] * len(pic_list)
            data_list.extend(pic_list)
            label_list.extend(lb_list)
    return folders, (data_list, label_list)


def train(model, batch_size, epochs, learning_rate, train_ds, test_ds, test_labels, labels):
    checkpoint_best = ModelCheckpoint(os.path.join("save", "best_epoch_weights.h5"),
                                      monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    # opt = tf.optimizers.SGD(learning_rate, momentum=0.9, weight_decay=0.0001)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.0001, momentum=0.9)
    model.compile(optimizer=opt,
                  metrics=["accuracy", AUC(), Recall(), Precision()],  
                  loss=[CategoricalCrossentropy()]) 
    history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, shuffle=True,
                        validation_data=(test_ds),
                        callbacks=[early_stopping, checkpoint_best])
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.save("save/model.ckpt")
    history = history.history

    evaluate_score = model.evaluate(train_ds)
    keys = list(history.keys())[5:]
    for i in range(5):
        print(f"{keys[i]}:{round(evaluate_score[i], 2)}")
    y_pred = model.predict(test_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_test = np.array(test_labels)
    cf = confusion_matrix(y_test, y_pred, labels=[i for i in range(len(labels))])
    plt.figure()
    ax = sns.heatmap(cf, annot=True, xticklabels=labels, yticklabels=labels)
    ax.tick_params(labelleft=True, labeltop=True, labelbottom=False)
    plt.title("Confusion matrix")
    plt.savefig("Confusion matrix")
    plt.close()
    for key in [["accuracy", "val_accuracy"], ["loss", "val_loss"]]:
        plt.figure()
        for k in key:
            plt.plot([i for i in range(len(history[k]))], history[k])
            tle = k.split("_")[-1]
            plt.title(tle)
        plt.savefig(tle)
        plt.close()


def main(model_name="resnet"):
    batch_size = 24  # batch size
    epochs = 30  
    learning_rate = 0.0005  
    labels, train_ds = read_datas("datasets/train")
    # print(train_ds)
    print(f"{len(train_ds[0])}")
    _, test_ds = read_datas("datasets/test")
    print(f"{len(test_ds[0])}")
    labels = [lb.split("-")[0] for lb in labels]
    with open("save/classes.txt", mode="w+", encoding="utf-8") as f:
        f.writelines(" ".join(labels))
    test_labels = test_ds[1]
    train_ds = load_ds(*train_ds,batch_size)
    test_ds = load_ds(*test_ds,batch_size)
   
    input_shape = 224
    model = CnnModel(len(labels), input_shape, model_name=model_name)  
    # if os.path.isfile("save//best_epoch_weights.h5"):
    #     print("loading weigts from save//best_epoch_weights.h5")
    #     model.build(input_shape=(None, input_shape, input_shape, 3))
    #     model.load_weights("save//best_epoch_weights.h5")
    train(model, batch_size, epochs, learning_rate, train_ds, test_ds, test_labels, labels)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  
    main("resnet")



