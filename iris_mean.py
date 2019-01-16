from sklearn.datasets import load_iris
from sklearn import cluster
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd


iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

e = cluster.KMeans(n_clusters=3)
e.fit(iris_df)

#iris_df.to_csv('iris_df.csv',encoding="SHIFT-JIS")

print(iris_df)
print(e.labels_)
print(e.cluster_centers_)

plt.scatter(iris_df.iloc[:,0],iris_df.iloc[:,2],marker="o",c=e.labels_,edgecolor="k")
plt.scatter(e.cluster_centers_[:,0],e.cluster_centers_[:,2],marker="x")
plt.show()
