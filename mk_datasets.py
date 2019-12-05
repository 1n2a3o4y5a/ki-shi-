
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#x:ノイズ調整
#y:seed
#z:分散
def fun(x,y,z):
    df = pd.DataFrame()
    for i in range(0,x,int(x / 10)):
        np.random.seed(y)
        array1 = np.random.normal(0,1,(10000,3))
        array2 = np.random.normal(0,1,(10000,3))
        array3 = np.random.normal(0,z,(1,3))
        array4 = np.random.normal(0,z,(i,3))
        array2[:i] = array2[:i] + array4
        array2[i:501] = array2[i:501] + array3
        experiment = array2
        #実験群をデータフレーム化
        target = pd.DataFrame(np.zeros(10000))
        target[:500] = 1
        target[501:] = 0
        df_experiment = pd.DataFrame(experiment)
        df = df.append(df_experiment)
        return df




# In[16]:


#import mk_datasets
import pprint
pprint.pprint(mk_datasets)

