from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


def fun(x,y,z):
    df = pd.DataFrame()
    np.random.seed(y)
    array1 = np.random.normal(0,1,(10000,3))
    array2 = np.random.normal(0,1,(10000,3))
    array3 = np.random.normal(0,z,(1,3))
    array4 = np.random.normal(0,z,(x,3))
    array2[:x] = array2[:x] + array4
    array2[x:501] = array2[x:501] + array3
    experiment = array2

    target = pd.DataFrame(np.zeros(10000))
    target[:500] = 1
    target[501:] = 0
    df_experiment = pd.DataFrame(experiment)
    df_experiment["target"] = target
    df = df.append(df_experiment)
    return df


r = []
f1score = []
for j in range(0,500,50):
    f1 = []
    R =
    for i in range(100):
        data = fun(j,i,24)
        X = data
        y = data["target"]
        model = Sequential()
        early_stopping = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 2, mode = "auto")
        model.add(Dense(100, input_dim=3))
        model.add(Activation('relu'))
        model.add(Dense(2, input_dim=100))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(X.iloc[:,0:3], y, nb_epoch=10000, batch_size=200,callbacks=[early_stopping],verbose = 0,validation_split = 0.1)
        data1 = fun(j,200 + i,24)
        X1 = data1
        y1 = data1["target"]
        predict = model.predict_classes(X.iloc[:,0:3])
        #print(predict)
        matrix = confusion_matrix(y,predict)



        TP = matrix[1,1]
        FP = matrix[0,1]
        FN = matrix[1,0]
        P = TP + FN
        if TP + FP == 0:
            precision = 0
        else:
            precision = float(TP) / (TP + FP)
        if P == 0:
            recall = 0
        else:
            recall = float(TP) / P
        if precision + recall == 0:
            F1 = 0
        else:
            F1 =float( 2 * precision * recall) / (precision + recall)

        R.append(j)
        f1.append(F1)
    f1score.append(f1)
    r.append(j)

print(f1)

fig, ax = plt.subplots()
bp = ax.boxplot(f1score)
plt.xticks([i for i in range(1,12)], r)
plt.setp(bp['medians'][0], color='red', linewidth=2)
plt.ylim([0,1.01])
plt.show()
