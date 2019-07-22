import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#有理ルンゲクッタ法

dt0 = 2
t = 10
N = 100
ye=2*np.exp(t)
ldt = [0]
a = [0]

for i in range(5):
        N = 10*N
        dt = t/N
        ldt.append(dt)
        y = 2
        for n in range(N):
            k1 = dt*y
            yn = y + k1 * 1 / 2
            k2 = dt * yn
            k3 = 2 * k1 - k2
            g11 = k1 ** 2
            g13 = k1 * k3
            g33 = k3 ** 2
            ynn = y + (2 * g13 * k1 - g11 * k3) / g33
            y = ynn
        k = np.abs(ynn-ye)/np.abs(ye)
        a.append(k)


plt.scatter(np.log10(ldt),np.log10(a),marker=".")
plt.grid(True)
plt.show()
