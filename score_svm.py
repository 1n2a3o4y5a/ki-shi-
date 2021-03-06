from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


np.random.seed(0)
array1 = np.random.normal(0,1,(10000,3))
array2 = np.random.normal(0,1,(10000,3))
array3 = np.random.normal(0,24,(501,3))
array2[:501] = array2[:501] + array3
experiment = array2
#実験群をデータフレーム化
target = pd.DataFrame(np.zeros(10000))
target[:500] = 1
target[501:] = 0
df_experiment = pd.DataFrame(experiment)
df_experiment["target"] = target
X = df_experiment
y = df_experiment["target"]

clf = svm.SVC()
clf.fit(X, y)

np.random.seed(1)
array1_1 = np.random.normal(0,1,(10000,3))
array2_1 = np.random.normal(0,1,(10000,3))
array3_1 = np.random.normal(0,24,(501,3))
array2_1[:501] = array2_1[:501] + array3_1
experiment_1 = array2_1
#実験群をデータフレーム化
target_1 = pd.DataFrame(np.zeros(10000))
target_1[:500] = 1
target_1[501:] = 0
df_experiment_1 = pd.DataFrame(experiment_1)
df_experiment_1["target"] = target
X2 = df_experiment_1
y2 = df_experiment_1["target"]

predict = clf.predict(X2)
score = accuracy_score(y2,predict)
print(score)
matrix = confusion_matrix(df_experiment["target"],predict)
print(matrix)

TP = matrix[1,1]
FP = matrix[0,1]
FN = matrix[1,0]
P = TP + FN
precision = TP / (TP + FP)
recall = TP / P
F1 = 2 / (1 / precision + 1 / recall)
print(F1)
print(recall)
