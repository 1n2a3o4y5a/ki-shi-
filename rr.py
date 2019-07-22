import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#オイラー法
dt0 = 2
t = 10
N = 100
ldt = [0]
y1 = [0]
ldt2 = [0]
z1 = [0]
rt = [0]
N = 10*N
dt = t/N


y = 1
z = 0
for n in range(N):
    yn = y + dt*z
    zn = z - dt*y
    y = yn
    z = zn
    z1.append(z)
    ldt2.append(n)
    y1.append(y)
    E = y **2 + z **2
    r = np.abs(E - 1) / 1
    rt.append(r)


plt.scatter(y1,z1)
plt.grid(True)
plt.show()
plt.scatter(ldt2,rt)
plt.grid(True)
plt.show()

#修正オイラー法
y = 1
z = 0
ldt2 = [0]
rt = [0]
y1 = [0]
z1 = [0]

for n in range(N):
    y2 = y + dt * z
    z2 = z - dt * y
    yn = y + ((z + z2)/2) * dt
    zn = z + ((-y - y2)/2) *dt
    y = yn
    z = zn
    ldt2.append(n)
    z1.append(z)
    y1.append(y)
    E = y **2 + z **2
    r = np.abs(E - 1) / 1
    rt.append(r)



plt.scatter(ldt2,rt)
plt.ylim(-0.00001,0.00001)
plt.grid(True)
plt.show()

#シンプレクティック法
dt0 = 2
t = 10
N = 100
ldt = [0]
y1 = [0]
ldt2 = [0]
z1 = [0]
rt = [0]
N = 10*N
dt = t/N


y = 1
z = 0
for n in range(N):
    yn = y + dt*z
    zn = (1 - dt **2) * z - dt*y
    y = yn
    z = zn
    z1.append(z)
    ldt2.append(n)
    y1.append(y)
    E = y **2 + z **2
    r = np.abs(E - 1) / 1
    rt.append(r)


plt.scatter(y1,z1)
plt.grid(True)
plt.show()
plt.scatter(ldt2,rt)
plt.grid(True)
plt.show()


#リープフロッグ法
t0 = 2
t = 10
N = 100
ldt = [0]
y1 = [0]
ldt2 = [0]
z1 = [0]
rt = [0]
N = 10*N
dt = t/N

y = 1
z = 0
yn = y + dt * z
zn = z - dt * y
for n in range(N):
    yn2 = y + 2 * dt * zn
    zn2 = z - 2 * dt * yn
    y = yn
    z = zn
    yn = yn2
    zn = zn2
    z1.append(z)
    ldt2.append(n)
    y1.append(y)
    E = y **2 + z ** 2
    r = np.abs(E - 1) / 1
    rt.append(r)

plt.scatter(y1,z1)
plt.grid(True)
plt.show()
plt.scatter(ldt2,rt)
plt.ylim(-0.00005,0.00005)
plt.grid(True)
plt.show()
