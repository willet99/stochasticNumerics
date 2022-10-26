import numpy as np
from Exercise1 import wiener_process
from Exercise3 import riemann_stieltjes
import matplotlib.pyplot as plt


def euler_maruyama(x0, T, N):
    [t,W] = wiener_process(T, N)
    h = T/N
    X = [x0]
    for n in range(N):
        f = [i*0.5 for i in X]
        G = X
        new_X = X[n] + h*f[n] + G[n]*(W[n+1]-W[n])
        X += [new_X]
    expl = np.exp(W)
    return [t,X,expl]


def milstein(x0, T, N):
    [t,W] = wiener_process(T, N)
    h = T/N
    X = [x0]
    for n in range(N):
        f = [i*0.5 for i in X]
        G = X
        new_X = X[n] + h*f[n] + G[n]*(W[n+1]-W[n]) + 0.5*G[n]**2*((W[n+1]-W[n])**2 - h)
        X += [new_X]
    expl = np.exp(W)
    return [t, X, expl]


def bdf2_maruyama(x0, T, N):
    [t,W] = wiener_process(T, N)
    h = T/N
    X = [x0, x0 + (h/2)*x0 + x0*(W[1]-W[0])]
    for n in range(2, N):
        f = [i*0.5 for i in X]
        G = X
        X_part = (4/3)*X[n-1] - (1/3)*X[n-2]
        # f_part = (2*h/3)*f[n]
        G_part = G[n-1]*(W[n]-W[n-1]) - (1/3)*G[n-2]*(W[n-1]-W[n-2])
        new_X = (X_part + G_part)/(1-(h/3))
        X += [new_X]
    expl = np.exp(W)
    return [t, X, expl]


def square_diff(X,expl):
    # For mean square difference, it would be good to run this
    # a couple of times for different paths and then take the empirical unbiased mean.
    diff = []
    for i in range(len(X)):
        diff += [(X[i]-expl[i])**2]
    return diff


[t,Xem,expl] = euler_maruyama(1,1,100)
plt.plot(t,Xem)
plt.plot(t,expl)
plt.show()
# plt.plot(t,square_diff(Xem,expl))
# plt.show()

[t,Xm,expl] = milstein(1,1,100)
plt.plot(t,Xm)
plt.plot(t,expl)
plt.show()
# plt.plot(t,square_diff(Xm,expl))
# plt.show()

[t,Xb,expl] = bdf2_maruyama(1,1,100)
plt.plot(t[:-1], Xb)
plt.plot(t[:-1], expl[:-1])
plt.show()
# plt.plot(t[:-1],square_diff(Xb,expl[:-1]))
# plt.show()