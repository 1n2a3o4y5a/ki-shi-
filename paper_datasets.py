import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#乱数の作成
np.random.seed(0)
array1 = np.random.normal(0,1,(10000,3))
array2 = np.random.normal(0,1,(10000,3))
array3 = np.random.normal(0,24,(1,3))
array2[:501] = array2[:501] + array3
experiment = array2

#実験群をデータフレーム化
target = pd.DataFrame(np.zeros(10000))
target[:500] = 1
target[501:] = 0
df_experiment = pd.DataFrame(experiment)
df_experiment["target"] = target

#ヒストグラム
plt.subplot(3,1,1)
plt.title("array1")
plt.hist(array1)
plt.subplot(3,1,2)
plt.title("array3")
plt.hist(array3)
plt.subplot(3,1,3)
plt.title("experiment")
plt.hist(experiment)
#plt.show()

#k-means
e = cluster.KMeans(n_clusters=2)
e.fit(experiment)
predict = KMeans(n_clusters=2).fit_predict(df_experiment)
pre = []
for i in predict:
    pre.append(i)
matrix = confusion_matrix(df_experiment["target"].values.tolist(), pre)
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
