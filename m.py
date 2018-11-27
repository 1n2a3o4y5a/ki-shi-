import numpy as np

np.random.seed(0)
a = np.random.randn(10000)

a_mean_np = np.mean(a)
a_var_np = np.var(a)

sum_i=0

for i in range(10000):
    sum_i=sum_i+a[i]

a_mean = sum_i*0.0001


print(a)




print(a_mean_np)
#print(a_var_np)
print(a_mean)
