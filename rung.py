#ルンゲクッタ法
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
        ldt.append(dt)
        y = 2
        for n in range(N):
            k1 = dt*y
            yn = y + k1 * 1 / 2
            k2 = dt * yn
            yn2 = y + k2 * 1/2
            k3 = dt * yn2
            yn3 = y + k3
            k4 = dt * yn3
            ynn = y + (k1 + 2 * k2 + 2 * k3 + k4)/6
            y = ynn

        k = np.abs(ynn-ye)/np.abs(ye)
        a.append(k)

plt.scatter(np.log10(ldt),np.log10(a),marker=".")
plt.grid(True)
plt.show()
