import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
array1 = np.random.normal(0,1,(10000,3))
array2 = np.random.normal(0,1,(10000,3))
array3 = np.random.normal(0,24,(1,3))
array2[:501] = array2[:501] + array3
experiment = array2

plt.subplot(3,1,1)
plt.title("array1")
plt.hist(array1)
plt.subplot(3,1,2)
plt.title("array3")
plt.hist(array3)
plt.subplot(3,1,3)
plt.title("experiment")
plt.hist(experiment)
plt.show()

heat_map = np.corrcoef(experiment[:501])
heat_map2 = np.mean(heat_map - np.diag(np.diag(heat_map)))
print(heat_map2)
