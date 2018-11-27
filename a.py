import matplotlib.pyplot as plt
import numpy as np

def f(x,w):
    return (x-w)*x*(x+2)


x = np.linspace(-3,3,100)

plt.subplot(1,2,1)
plt.plot(x,f(x,2),color="black",label="w=2")
plt.legend(loc="upper left")
plt.title("f(x)")
plt.ylim(-15,15)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(x,f(x,1),color="cornflowerblue",label="w=1")

plt.legend(loc="upper left")
plt.title("f(x)")
plt.ylim(-15,15)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
