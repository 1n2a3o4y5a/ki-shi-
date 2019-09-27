import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

np.random.seed(0)
array1 = np.random.normal(0,1,(10000,3))
array2 = np.random.normal(0,1,(10000,3))
array3 = np.random.normal(0,24,(1,3))
array4 = np.random.normal(0,24,(200,3))
array2[:251] = array2[:251] + array3
array2[251:501] = array2[251:501] + array3
experiment = array2
#実験群をデータフレーム化
target = pd.DataFrame(np.zeros(10000))
target[:500] = 1
target[501:] = 0
df_experiment = pd.DataFrame(experiment)
df_experiment["target"] = target

X = df_experiment
y = df_experiment["target"]
nn = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
nn.fit(X,y)

np.random.seed(1)
array1_1 = np.random.normal(0,1,(10000,3))
array2_1 = np.random.normal(0,1,(10000,3))
array3_1 = np.random.normal(0,24,(1,3))
array4_1 = np.random.normal(0,24,(200,3))
array2_1[:251] = array2_1[:251] + array3_1
array2_1[251:501] = array2_1[251:501] + array3_1
experiment_1 = array2_1
#実験群をデータフレーム化
target_1 = pd.DataFrame(np.zeros(10000))
target_1[:500] = 1
target_1[501:] = 0
df_experiment_1 = pd.DataFrame(experiment_1)
df_experiment_1["target"] = target_1
X2 = df_experiment_1
y2 = df_experiment_1["target"]


print (nn.score(df_experiment, df_experiment["target"]))

predict = nn.predict(df_experiment_1)
matrix = confusion_matrix(df_experiment_1["target"],predict)
print(matrix)

TP = matrix[1,1]
FP = matrix[0,1]
FN = matrix[1,0]
P = TP + FN
precision = TP / (TP + FP)
recall = TP / P
F1 = 2 / (1 / precision + 1 / recall)
print(F1)
