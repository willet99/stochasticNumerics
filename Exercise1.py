import random

import numpy as np
import numpy.linalg as nl
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

n = 1000  # no realizations
T = 1   # end of time interval
std = 1     # standard deviation of wiener process, needed?
N = 100     # points to simulate


def wiener_process(T, N, std=1, time=True):
    h = T / N  # time step (equidistant)

    Z = [np.random.normal(0, std**2) for i in range(N + 1)]  # Vector of normal distributed numbers

    W_0 = 0

    W = [W_0]

    for i in range(len(Z) - 1):
        k = i + 1
        W += [W[k - 1] + np.sqrt(h) * Z[k]]

    if time:
        t = [h * i for i in range(N + 1)]  # discretisation time
        return [t, W]
    else:
        return W


def plot_wiener(T, N, n=10, std=1, mean=True, var=True, plot_all=True):
    # mu_temp = [0 for i in range(N+1)]
    x = []
    wien_temp = wiener_process(T, N, std)
    t = wien_temp[0]
    for j in range(n):
        x_temp = wiener_process(T, N, std, time=False)
        x += [x_temp]

    s_x = pd.DataFrame(x)
    if plot_all:
        s_x.apply(lambda x: plt.plot(t, x), axis=1)
    if var:
        wiener_var = s_x.var(0)
        if (wiener_var.values < 0).any():
            print('Something wrong with variance. Negative values.')
        else:
            plt.plot(t, wiener_var, linestyle='dotted', label='Variance')
    if mean:
        wiener_mean = s_x.mean(0)
        plt.plot(t, wiener_mean, linestyle='dashed', label='Mean')
    if var or mean:
        plt.legend()
    plt.ylabel('W(t)')
    plt.xlabel('t [s]')
    plt.show()


def pdf(x):
    if isinstance(x, (int, float)):
        x = np.array([x])
        one_val = True
    else:
        one_val = False
    if (x < 0).any() or (x > 1).any():
        print('x not in the right range')
    f = [np.exp(y)/(np.exp(1)-1) for y in x]
    if one_val:
        f = f[0]
    return f


def acceptance_rejection(n):
    c = pdf(1)
    i = 0
    x = []
    while i < n:
        [U, V] = np.random.uniform(0, 1, 2)
        if U < pdf(V)/c:
            x += [V]
            i += 1
    return x


def inverse_transform(n):
    return None


# X = acceptance_rejection(n)
# fig, ax = plt.subplots()
# temp = ax.hist(X, density=True)
# x = np.linspace(0, 1, 1000)
# f_x = pdf(x)
# ax.plot(x, f_x)
# plt.show()


# plot_wiener(T, N, n, std)
# plot_wiener(T, N, n, std, plot_all=False)
# plot_wiener(T, N, n, std, mean=False, var=False)







