import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np

a=pd.read_csv("wind_data_copy.csv",encoding="SHIFT-JIS",usecols=[2,5])
a2=a.dropna(axis=0)

x=a2.iloc[:,[0]].values.tolist()
y=a2.iloc[:,[1]].values.tolist()

#print(a2)
mean_x=np.mean(x)
mean_y=np.mean(y)
var_x=np.var(x)
var_y=np.var(y)

sum_s=0
sum_k=0

for s in range(len(a2)):
    sum_s=sum_s+(mean_x-x[s])**3

v3_x=np.sqrt(len(a2))*sum_s/np.sqrt(len(a2)*var_x)**3

for k in range(len(a2)):
    sum_k=sum_k+(mean_x-x[k])**4

v4_x=len(a2)*sum_k/(len(a2)*var_x)**2

sum_s=0
sum_k=0

for s in range(len(a2)):
    sum_s=sum_s+(mean_y-y[s])**3

v3_y=np.sqrt(len(a2))*sum_s/np.sqrt(len(a2)*var_y)**3

for k in range(len(a2)):
    sum_k=sum_k+(mean_y-y[k])**4

v4_y=len(a2)*sum_k/(len(a2)*var_y)**2


x=pd.DataFrame(x)
y=pd.DataFrame(y)

x.hist()
plt.title(str(mean_x.round(3))+"/"+str(var_x.round(3))+"/"+str(v3_x.round(3))+"/"+str(v4_x.round(3)))
plt.show()
y.hist()
plt.title(str(mean_y.round(3))+"/"+str(var_y.round(3))+"/"+str(v3_y.round(3))+"/"+str(v4_y.round(3)))
plt.show()


corr=a2.corr()
print(corr)
plt.plot(x,y)
plt.title("corrcoef="+str(corr))
plt.show()

np.random.seed(0)
ran=np.random.rand(len(a2))
c=list(range(len(a2)))

for i in range(100):
    np.random.seed(i)
    ran=np.random.rand(len(a2))

    df_a=pd.DataFrame({"c":c,
                  "ran":ran})
    df_b=df_a.sort_values(by=["ran"])
    df_b2=df_b.assign(D=c)

    z=df_b2.corr()

    if np.abs(-0.578651)>np.abs(z.iloc[0,2]):
        print("true")
    else:
        break
    #print(z.iloc[0,2])
