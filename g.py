import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


np.random.seed(0)
x_min=0
x_max=2.5
x_n=30
x_col=["cornflowerblue","gray"]
X=np.zeros(x_n)
T=np.zeros(x_n,dtype=np.uint8)
Dist_s=[0.4,0.8]
Dist_w=[0.8,1.6]
Pi=0.5

for n in range(x_n):
    wk=np.random.rand()
    T[n]=0*(wk<Pi)+1*(wk>=Pi)
    X[n]=np.random.rand()*Dist_w[T[n]]+Dist_s[T[n]]

def show_data1(x,t):
    K=np.max(t)+1
    for k in range(K):
        plt.plot(X[t==k],t[t==k],x_col[k],alpha=0.5,linestyle="none",marker="o")
    plt.grid(True)
    plt.ylim(-0.5,1.5)
    plt.xlim(x_min,x_max)
    plt.yticks([0,1])

fig=plt.figure(figsize=(3,3))
show_data1(X,T)

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

N=100
K=3
T3=np.zeros((N,3), dtype=np.uint8)
T2=np.zeros((N,2),dtype=np.uint8)
X=np.zeros((N,2))
X_range0=[-3,3]
X_range1=[-3,3]
Mu=np.array([[-.5,-.5],[.5,1.0],[1,-.5]])
Sig=np.array([[.7,.7],[.8,.3],[.3,.8]])
Pi=np.array([0.4,0.8,1])
for n in range(N):
    wk=np.random.rand()
    for k in range(K):
        if wk<Pi[k]:
            T3[n,k]=1
            break
    for k in range(2):
        X[n,k]=(np.random.randn()*Sig[T3[n,:]==1,k]+Mu[T3[n,:]==1,k])
T2[:,0]=T3[:,0]
T2[:,1]=T3[:,1]|T3[:,2]


#3次元ロジスティック回帰
def logistic3(x0,x1,w):
    K=3
    w=w.reshape((3,3))
    n=len(x1)
    y=np.zeros((n,K))
    for k in range(K):
        y[:,k]=np.exp(w[k,0]*x0+w[k,1]*x1+w[k,2])
    wk=np.sum(y,axis=1)
    wk=y.T/wk
    return y

def cee_logistic3(w,x,t):
    X_n=X.shape[0]
    y=logistic3(x[:,0],x[:,1],w)
    cee=0
    N,K=y.shape
    for n in range(N):
        for k in range(K):
            cee=cee-(t[n,k]*np.log(y[n,k]))
    cee=cee/X_n
    return cee

def dcee_logistic3(w,x,t):
    X_n=X.shape[0]
    y=logistic3(x[:,0],x[:,1],w)
    dcee=np.zeros((3,3))
    N,K=y.shape
    for n in range(N):
        for k in range(K):
            dcee[k,:]=dcee[k,:]-(t[n,k]-y[n,k])*np.r_[X[n,:],1]
    dcee=dcee/X_n
    return dcee.reshape(-1)

def fit_logistic3(w_init,x,t):
    res=minimize(cee_logistic3,w_init,args=(x,t),jac=dcee_logistic3,method="CG")
    return res.x

W_init=np.zeros((3,3))
W=fit_logistic3(W_init,X,T3)
print(np.round(W.reshape((3,3)),2))
cee=cee_logistic3(W,X,T3)
print("CEE={0:.2f}".format(cee))

plt.figure(figsize=(3,3))
#show_data2(X,T3)
#show_contour_logistic3(W)
plt.show()
