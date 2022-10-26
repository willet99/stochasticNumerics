import numpy as np
from Exercise1 import wiener_process
import matplotlib.pyplot as plt


def geometric_brownian_motion(x0, mu, sigma, T, N):
    [t, W] = wiener_process(T, N)
    t = np.array(t)
    W = np.array(W)
    exponent = (mu-sigma**2/2)*t + sigma*W
    X = x0*np.exp(exponent)
    return t, X


# a)
def ex2a():
    for i in range(20):
        [t, aX] = geometric_brownian_motion(i, 0.1, 0.5, 5, 100)
        plt.plot(t, aX)
    plt.show()


# b)
def ex2b():
    for i in range(20):
        [t, bX] = geometric_brownian_motion(i, 0.1, 0.2, 5, 100)
        plt.plot(t, bX)
    plt.show()


def brownian_motion_on_circle(x0, T, N):
    [t, W] = wiener_process(T, N)
    t = np.array(t)
    W = np.array(W)
    x0 = np.array(x0)
    x0 = x0.transpose()
    phi = np.array([np.cos(W)*x0[0] - np.sin(W)*x0[1], np.sin(W)*x0[0] + np.cos(W)*x0[1]])
    return [t, phi]


def plot_brownian_motion_on_circle():
    [t, cX] = brownian_motion_on_circle([1, 0], 10, 100)
    for ts in t:
        plt.plot(cX[0], cX[1])
    plt.show()


def riemann_stieltjes(a, T, N):
    [t,W] = wiener_process(T,N)
    Sn = 0
    for i in range(N):
        f = (1-a)*W[i] + a*W[i+1]
        dw = W[i+1] - W[i]
        Sn = Sn + f*dw
    return Sn


def comparison_rs(a,T):
    N=2
    [t,W] = wiener_process(T,N)
    return 0.5*W[-1]**2 + (a-0.5)*t[-1]


def compare_mean(a, T, N, n):
    comp_rs = a*T
    snsum = 0
    for i in range(n):
        rs = riemann_stieltjes(a, T, N)
        snsum = snsum + rs
    return snsum/n


# a_choice = [0, 0.25, 0.5]
#
# T=5
# N=100
# n=10000
#
# diff = []
# diff_mean = []
# for a in a_choice:
#     diff_mean += [compare_mean(a, T, N, n)]
#
# print(diff_mean)