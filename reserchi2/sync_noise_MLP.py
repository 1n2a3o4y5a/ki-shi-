#同期ノイズを乗せたMLP

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

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
    array = np.random.normal(0,z,(1,3))
    array5 = array.copy()
    array5[0,1:2] = 0
    array6 = array.copy()
    array6[0,0] = 0
    array6[0,2] = 0
    array7 = array.copy()
    array7[0,0:1] = 0
    experiment[:200] = experiment[:200] + array5
    experiment[200:400] = experiment[200:400] + array6
    experiment[400:600] = experiment[400:600] + array7
    #実験群をデータフレーム化
    target = pd.DataFrame(np.zeros(10000))
    target[:500] = 1
    target[501:] = 0
    df_experiment = pd.DataFrame(experiment)
    df_experiment["target"] = target
    df = df.append(df_experiment)
    return df

f1score = []
for i in range(100):
    data = fun(100,i,24)
    X = data
    y = data["target"]
    nn = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
    nn.fit(X,y)
    data1 = fun(100,200 + i,24)
    X1 = data1
    y1 = data1["target"]
    predict = nn.predict(X1)
    matrix = confusion_matrix(y1,predict)
    TP = matrix[1,1]
    FP = matrix[0,1]
    FN = matrix[1,0]
    P = TP + FN
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / P
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * precision * recall / (precision + recall)
    print(F1)
    f1score.append(F1)

fig, ax = plt.subplots()
bp = ax.boxplot(f1score)
plt.setp(bp['medians'][0], color='red', linewidth=2)
plt.ylim([0,1.01])
plt.grid()
plt.show()


# In[30]:
