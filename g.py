import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
x_min=0
x_max=2.5
x_n=30
x_col=["cornflowerblue","gray"]
x=np.zeros(x_n)
T=np.zeros(x_n,dtype=np.uint8)
Dist_s=[0.4,0.8]
Dist_w=[0.8,1.6]
Pi=0.5

for n in range(x_n):
    wk=np.random.rand()
    T[n]=0*(wk<Pi)+1*(wk>=Pi)
    x[n]=np.random.rand()*Dist_w[T[n]]+Dist_s[T[n]]

def show_data1(x,t):
    K=np.max(t)+1
    for k in range(K):
        plt.plot(x[t==k],t[t==k],x_col[k],alpha=0.5,linestyle="none",marker="o")
    plt.grid(True)
    plt.ylim(-0.5,1.5)
    plt.xlim(x_min,x_max)
    plt.yticks([0,1])

fig=plt.figure(figsize=(3,3))
show_data1(x,T)

def logistic(x,w):
    y=1/(1+np.exp(-(w[0]*x+w[1])))
    return y

def show_logistic(w):
    xb=np.linspace(x_min,x_max,100)
    y=logistic(xb,w)
    plt.plot(xb,y,color="gray",linewidth=4)
    i=np.min(np.where(y>0.5))
    B=(xb[i-1]+xb[i])/2
    plt.plot([B,B],[-.5,1.5],color="k",linestyle='--')
    plt.grid(True)
    return B

def cee_logistic(w,x,t):
    y=logistic(x,w)
    cee=0
    for n in range(len(y)):
        cee=cee-(t[n]*np.log(y[n])+(1-t[n])*np.log(1-y[n]))
        cee=cee/x_n
        return cee

w=[8,-10]
show_logistic(w)
plt.show()
