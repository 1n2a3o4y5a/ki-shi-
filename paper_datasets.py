import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#乱数の作成
<<<<<<< HEAD
=======
i = 100
>>>>>>> 94a7a04b363829cf6d9f0ed98f5a343b74d99b7e
np.random.seed(0)
array1 = np.random.normal(0,1,(10000,3))
array2 = np.random.normal(0,1,(10000,3))
array3 = np.random.normal(0,24,(1,3))
<<<<<<< HEAD
array2[:501] = array2[:501] + array3
experiment = array2

=======
array4 = np.random.normal(0,240,(i,3))
array2[:i] = array2[:i] + array4
array2[i:501] = array2[i:501] + array3
experiment = array2
>>>>>>> 94a7a04b363829cf6d9f0ed98f5a343b74d99b7e
#実験群をデータフレーム化
target = pd.DataFrame(np.zeros(10000))
target[:500] = 1
target[501:] = 0
df_experiment = pd.DataFrame(experiment)
df_experiment["target"] = target

<<<<<<< HEAD
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
=======


#k-means
e = cluster.KMeans(n_clusters=2,random_state=0)
e.fit(experiment)

np.random.seed(1)
array1_1 = np.random.normal(0,1,(10000,3))
array2_1 = np.random.normal(0,1,(10000,3))
array3_1 = np.random.normal(0,24,(1,3))
array4_1 = np.random.normal(0,24,(i,3))
array2_1[:i] = array2_1[:i] + array4_1
array2_1[i:501] = array2_1[i:501] + array3_1
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
for i in pre:
    predict.append(i)
matrix = confusion_matrix(df_experiment_1["target"].values.tolist(), predict)
>>>>>>> 94a7a04b363829cf6d9f0ed98f5a343b74d99b7e
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
