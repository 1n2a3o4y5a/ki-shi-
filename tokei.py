import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
a = np.random.randn(10000)

plt.hist(a)
plt.show()


a_mean_np = np.mean(a)
a_var_np = np.var(a)
sum_i=0
sum_j=0
sum_s=0
sum_k=0

for i in range(10000):
    sum_i=sum_i+a[i]

a_mean = sum_i*0.0001

for n in range(10000):
    sum_j=sum_j+(a_mean-a[n])**2

a_var = sum_j*0.0001

for s in range(10000):
    sum_s=sum_s+(a_mean-a[s])**3

v3=100*sum_s/np.sqrt(sum_j)**3

for k in range(10000):
    sum_k=sum_k+(a_mean-a[k])**4

v4=10000*sum_k/sum_j**2


print(a_mean_np)
print(a_var_np)
print(a_mean)
print(a_var)
print(v3)
print(v4)
