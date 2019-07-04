import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

def make(x,i):
    np.random.seed(i)
    array1 = np.random.normal(0,1,(10000,3))
    array2 = np.random.normal(0,1,(10000,3))
    array3 = np.random.normal(0,24,(x,3))
    array4 = np.random.normal(0,24,(500 - x,3))
    array2[:501] = array2[:501] + array3
    experiment = array2

    #実験群をデータフレーム化
    target = pd.DataFrame(np.zeros(10000))
    target[:500] = 1
    target[501:] = 0
    df_experiment = pd.DataFrame(experiment)
    df_experiment["target"] = target
    print(df_experiment)
