import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import svm


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
    return df

r = []
f1score = []
for i in range(1):
    f1 = []
    for j in range(0,500,50):
        data = fun(j,i,24)
        X = data
        y = data["target"]
        clf = svm.SVC()
        clf.fit(X, y)
        data1 = fun(j,200 + i,24)
        X1 = data1
        y1 = data1["target"]
        predict = clf.predict(X1)
        matrix = confusion_matrix(y1,predict)
        #print(matrix)
        F1 = f1_score(y1,predict)
        #print(f1_score(y1,predict))
        r.append(j)
        f1.append(F1)
    f1score.append(f1)


print(f1score)
print(len(r))
ans = pd.DataFrame(r)
for i in range(len(f1score)):
    ans = pd.concat([ans,pd.Series(f1score[i])],axis=1)
print(ans)
plt.scatter(r,f1score)
plt.show()
