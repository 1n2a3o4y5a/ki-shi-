import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


a = pd.read_csv("wind_data_copy.csv",encoding="SHIFT-JIS")
kiatsu = a.iloc[:,2].values
kion = a.iloc[:,5].values


plt.hist(kiatsu,range(950,1050))
plt.show()
plt.hist(kion,range(-15,40))
plt.show()


plt.scatter(kiatsu,kion)
plt.show()
