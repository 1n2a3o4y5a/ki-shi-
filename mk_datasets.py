import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#x:ノイズ調整
#y:seed
#z:分散
def fun(x,y,z):
    df = pd
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
        df_experiment["target"] = target


fun(500,1,24)
