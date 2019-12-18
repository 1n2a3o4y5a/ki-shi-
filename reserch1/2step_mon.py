import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import svm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from statistics import mode

#x:ノイズ調整
#y:seed
#z:分散
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
    #実験群をデータフレーム化
    target = pd.DataFrame(np.zeros(10000))
    target[:500] = 1
    target[501:] = 0
    df_experiment = pd.DataFrame(experiment)
    df_experiment["target"] = target
    df = df.append(df_experiment)
    return experiment

r = []
f1score = []
for j in range(1):
    f1 = []
    R = []
    for i in range(10):
        data = fun(j,i,24)
        experiment = data
        X = data
        target = pd.DataFrame(np.zeros(10000))
        target[:500] = 1
        target[501:] = 0
        df_X = pd.DataFrame(X)
        df_X["target"] = target
        #実験群に対して中央絶対偏差を計算
        median_X = np.zeros((10000,3))
        for k in range(10000):
            X_median = np.median(X[k])
            median_X[k] = X_median
        experiment = np.abs(experiment - median_X)
        MAD1 = []
        for k in range(10000):
            mad = np.median(experiment[k])
            MAD1.append(mad)
        #対照群に対して中央絶対偏差を計算
        control = np.random.normal(0,1,(10000,3))
        median_cont = np.zeros((10000,3))
        for k in range(10000):
            cont_median = np.median(control[k])
            median_cont[k] = cont_median
        control2 = np.abs(control - median_cont)
        MAD2 = []
        for k in range(10000):
            mad = np.median(control2[k])
            MAD2.append(mad)

        #MAD
        A = pd.DataFrame()
        for k in range(10000):
            if MAD1[k] > MAD2[k] * 2:
                A = A.append(pd.Series(df_X.iloc[k]))
        result1 = linkage(A,
                          metric = "correlation",
                          method = "average")
        clustered = fcluster(result1, 0.75, criterion='distance')
        #plt.figure()
        #dendrogram(result1, orientation='right', labels=list(A.index), color_threshold=0.75)
        #plt.show()
        m = mode(clustered)
        TP = 0
        FP = 0

        for i, c in enumerate(clustered):
            if c == m and A.iloc[i].iloc[3] == 1:
                TP += 1
            elif c == m and A.iloc[i].iloc[3] != 0:
                FP += 1

        FN = 500 - TP
        TN = 9500 - FP
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        recall = TP / 500
        if precision + recall == 0:
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall)

        at = A["target"] == 1
        f1.append(F1)
        R.append(j)
    f1score.append(f1)
    r.append(R)

print(f1score)
fig, ax = plt.subplots()
bp = ax.boxplot(f1score)
plt.setp(bp['medians'][0], color='red', linewidth=2)
plt.show()
