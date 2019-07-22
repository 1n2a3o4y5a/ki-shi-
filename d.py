import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt0 = 2
t = 10
N = 100
ye=2*np.exp(t)
ldt = [0]
a = [0]

for i in range(5):
    N = 10*N
    dt = t/N
    #ldt = [dt]
    ldt.append(dt)
    y =2
    for n in range(N):
        yn = y + dt*y
        y = yn
    k = np.abs(yn-ye)/np.abs(ye)
    a.append(k)

print(yn)
print(a)



ldt.pop(0)
a.pop(0)

ldt2 = [0]
b = [0]
N = 100
for i in range(5):
    N = 10*N
    dt2 = t/N
    ldt2.append(dt2)
    y2 = 2
    for n in range(N):
        z = y2 + dt2*y2
        yn2 = y2 + ((y2+z)/2)*dt2
        y2 = yn2
    k2 = np.abs(yn2-ye)/np.abs(ye)
    b.append(k2)

print(yn2)
print(b)

ldt3 = [0]
c = [0]
N = 100
for i in range(5):
    N = 10*N
    dt3 = t/N
    ldt3.append(dt3)
    y1 = 2
    yn = 2 + dt3*2
    for n in range(N-1):
        yn3 = y1 + 2*dt3*yn
        y1 = yn
        yn = yn3
    k3 = np.abs(yn3-ye)/np.abs(ye)
    c.append(k3)

print(yn3)
print(c)

plt.scatter(np.log10(ldt),np.log10(a),marker=".")
plt.grid(True)
plt.scatter(np.log10(ldt2),np.log10(b),marker=".",color="black")
plt.grid(True)
plt.scatter(np.log10(ldt3),np.log10(c),marker="^")
plt.grid(True)
plt.show()
