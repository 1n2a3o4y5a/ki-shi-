import numpy as np
import pandas as pd

np.random.seed(0)
a=np.random.rand(20)
c=list(range(1,21))

for i in range(10):
    np.random.seed(i)
    a=np.random.rand(20)

    df_a=pd.DataFrame({"c":c,
                  "a":a})
    df_b=df_a.sort_values(by=["a"])
    df_b2=df_b.assign(D=c)

    z=df_b2.corr()
    print(z.iloc[0,2])
