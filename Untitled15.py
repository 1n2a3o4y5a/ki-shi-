
# coding: utf-8

# In[19]:


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
    for j in range(0,50,1):
        data = fun(j,i,24)
        X = data
        y = data["target"]
        nn = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
        nn.fit(X,y)
        data1 = fun(j,200 + i,24)
        X1 = data1
        y1 = data1["target"]
        predict = nn.predict(X1)
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


# In[20]:


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
    for j in range(0,50,1):
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


# In[23]:


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
    for j in range(0,50,1):
        data = fun(j,i,24)
        X = data
        y = data["target"]
        nn = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
        nn.fit(X,y)
        data1 = fun(j,200 + i,24)
        X1 = data1
        y1 = data1["target"]
        predict = nn.predict(X1)
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


# In[ ]:


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
for i in range(3):
    f1 = []
    R = []
    for j in range(0,50,1):
        data = fun(j,i,24)
        X = data
        y = data["target"]
        nn = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
        nn.fit(X,y)
        data1 = fun(j,200 + i,24)
        X1 = data1
        y1 = data1["target"]
        predict = nn.predict(X1)
        matrix = confusion_matrix(y1,predict)
        #print(matrix)
        F1 = f1_score(y1,predict)
        #print(f1_score(y1,predict))
        R.append(j)
        f1.append(F1)
    f1score.append(f1)
    r.append(R)


fig, ax = plt.subplots()
bp = ax.boxplot(f1score)
    

plt.show()


# In[ ]:


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data           
Y = iris.target 

model = Sequential()
model.add(Dense(12, input_dim=4))
model.add(Activation('relu'))
model.add(Dense(3, input_dim=12))
model.add(Activation('softmax')) 
model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)
model.fit(train_X, train_Y, nb_epoch=20, batch_size=5)

f1 = f1_score(test_Y,testx)
loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))

