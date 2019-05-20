import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score


score = []
tp = []
fp = []
fn = []

def fun(x):
    for i in range(x):
        np.random.seed(i)
        array1 = np.random.normal(0,1,(10000,3))
        array2 = np.random.normal(0,1,(10000,3))
        array3 = np.random.normal(0,8,(1,3))
        array2[:501] = array2[:501] + array3
        experiment = array2
        #実験群をデータフレーム化
        target = pd.DataFrame(np.zeros(10000))
        target[:500] = 1
        target[501:] = 0
        df_experiment = pd.DataFrame(experiment)
        df_experiment["target"] = target
        #k-means,matrix
        e = cluster.KMeans(n_clusters=2)
        e.fit(experiment)
        predict = KMeans(n_clusters=2).fit_predict(df_experiment)
        pre = []
        for i in predict:
            pre.append(i)
        matrix = confusion_matrix(df_experiment["target"].values.tolist(), pre)
        #print(matrix)
        #F1スコア
        if matrix[0,0] > matrix[0,1]:
            TP = matrix[1,1]
            FP = matrix[0,1]
            FN = matrix[1,0]
            P = TP + FN
            precision = TP / (TP + FP)
            recall = TP / P
            F1 = 2 / (1 / precision + 1 / recall)
            tp.append(TP)
            fp.append(FP)
            fn.append(FN)
            score.append(F1)

        else:
            TP = matrix[1,0]
            FP = matrix[0,0]
            FN = matrix[1,1]
            P = TP + FN
            precision = TP / (TP + FP)
            recall = TP / P
            F1 = 2 / (1 / precision + 1 / recall)
            tp.append(TP)
            fp.append(FP)
            fn.append(FN)
            score.append(F1)


fun(1000)
#print(score)
#print(tp)
#print(fp)
#print(fn)
plt.hist(score)
plt.show()
