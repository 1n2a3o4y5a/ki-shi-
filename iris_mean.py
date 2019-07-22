from sklearn.datasets import load_iris
from sklearn import cluster
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

e = cluster.KMeans(n_clusters=3)
e.fit(iris_df)
predict = KMeans(n_clusters=3).fit_predict(iris_df)
#print(iris_df["target"].values.tolist())

#iris_df.to_csv('iris_df.csv',encoding="SHIFT-JIS")

#print(iris_df)
#print(e.labels_)
#print(e.cluster_centers_)
pre = []
for i in predict:
    pre.append(i)

#print(pre)
#print(iris_df["target"].values.tolist())


plt.scatter(iris_df.iloc[:,0],iris_df.iloc[:,2],marker="o",c=e.labels_,edgecolor="k")
plt.scatter(e.cluster_centers_[:,0],e.cluster_centers_[:,2],marker="x")
plt.show()

matrix = confusion_matrix(iris_df["target"].values.tolist(), pre)
print(matrix)
