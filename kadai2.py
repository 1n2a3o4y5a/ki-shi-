import numpy as np
import pandas as pd

c=list(range(1,52561))

a = pd.read_csv("wind_data_copy.csv",encoding="SHIFT-JIS")
kiatsu = a.iloc[:,2].values.tolist()
kion = a.iloc[:,5].values.tolist()

for i in range(1,52561):
    np.random.seed(i)
    b=np.random.rand(52560)

    df_a=pd.DataFrame({"c":c,
                  "b":b})
    df_b=df_a.sort_values(by=["b"])
    df_b2=df_b.assign(D=c)

    z=df_b2.corr()
print(z.iloc[0,2])
