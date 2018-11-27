import numpy as np
import matplotlib.pyplot as plt

u1 = np.random.rand(1000)
u2 = np.random.rand(1000)

x1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
x2 = np.sqrt(-2*np.log(u2))*np.sin(2*np.pi*u1)

x1_mean = np.mean(x1)
x1_var =  np.var(x1)
x2_mean = np.mean(x2)
x2_var = np.var(x2)

plt.hist(x1,bins=50)
plt.show()
plt.hist(x2,bins=50)
plt.show()

print(x1,x2,x1_mean,x1_var,x2_mean,x2_var)
