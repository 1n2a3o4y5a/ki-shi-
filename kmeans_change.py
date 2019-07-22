import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

f1_score = []
for i in range(0,500,50):
    #乱数の作成
    np.random.seed(0)
    array1 = np.random.normal(0,1,(10000,3))
    array2 = np.random.normal(0,1,(10000,3))
    array3 = np.random.normal(0,24,(1,3))
    array4 = np.random.normal(0,24,(i,3))
    array2[:i + 1] = array2[:i + 1] + array3
    array2[i:501] = array2[i:501] + array4
    experiment = array2
    #実験群をデータフレーム化
    target = pd.DataFrame(np.zeros(10000))
    target[:500] = 1
    target[501:] = 0
    df_experiment = pd.DataFrame(experiment)
    df_experiment["target"] = target

    e = cluster.KMeans(n_clusters=2)
    e.fit(experiment)

    np.random.seed(1)
    array1_1 = np.random.normal(0,1,(10000,3))
    array2_1 = np.random.normal(0,1,(10000,3))
    array3_1 = np.random.normal(0,24,(1,3))
    array4_1 = np.random.normal(0,24,(i,3))
    array2_1[:i + 1] = array2_1[:i + 1] + array3_1
    array2_1[i:501] = array2_1[i:501] + array4_1
    experiment_1 = array2_1
    #実験群をデータフレーム化
    target_1 = pd.DataFrame(np.zeros(10000))
    target_1[:500] = 1
    target_1[501:] = 0
    df_experiment_1 = pd.DataFrame(experiment_1)
    df_experiment_1["target"] = target_1
    X2 = df_experiment_1
    y2 = df_experiment_1["target"]

    pre = KMeans(n_clusters=2).fit_predict(df_experiment_1)
    predict = []
    for j in pre:
        predict.append(j)
    matrix = confusion_matrix(df_experiment_1["target"].values.tolist(), predict)
    print(matrix)

    #Fスコア計算
    TP = matrix[1,1]
    FP = matrix[0,1]
    FN = matrix[1,0]
    P = TP + FN
    precision = TP / (TP + FP)
    recall = TP / P
    F1 = 2 / (1 / precision + 1 / recall)
    print(F1)
