import random
import math
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


def type_handler(x):
    if isinstance(x, (int, float)):
        x = np.array([x])
        one_val = True
    else:
        one_val = False
    return x, one_val


def check_range(x):
    range_ok = True
    if (x < 0).any() or (x > 1).any():
        range_ok = False
    return range_ok


def pdf(x):
    x, one_val = type_handler(x)
    range_ok = check_range(x)
    if not range_ok:
        print('x not in the right range')
        return
    f = [np.exp(y)/(np.exp(1)-1) for y in x]
    if one_val:
        f = f[0]
    return f


def cdf(x):
    return pdf(x)-pdf(0)


def inv_cdf(x):
    x, one_val = type_handler()
    range_ok = check_range(x)
    if not range_ok:
        print('x not in the right range')
        return
    G = [np.log((np.exp(1)-1)*y + 1) for y in x]
    if one_val:
        G = G[0]
    return G


def acceptance_rejection(N):
    c = pdf(1)
    i = 0
    x = []
    nr_it = 0
    while i < N+1:
        [U, V] = np.random.uniform(0, 1, 2)
        if U < pdf(V)/c:
            x += [V]
            i += 1
        nr_it += 1
    return x, nr_it


def acceptance_rejection_test(N, plot=True):
    X, nr_it = acceptance_rejection(N)
    if plot:
        fig, ax = plt.subplots()
        temp = ax.hist(X, density=True)
        x = np.linspace(0, 1, 1000)
        f_x = pdf(x)
        ax.plot(x, f_x)
        plt.show()
    else:
        print('Acceptance rate: ', (N + 1) / nr_it)


def inverse_transform(N):
    i = 0
    x = []
    while i < N+1:
        U = np.random.uniform(0, 1)
        x += [inv_cdf(U)]
        i += 1
    return x


def inverse_transform_test(N):
    X = inverse_transform(N)
    fig, ax = plt.subplots()
    temp = ax.hist(X, density=True)
    x = np.linspace(0, 1, 1000)
    f_x = pdf(x)
    ax.plot(x, f_x)
    plt.show()


def plot_realizations(N, n, acc_rej=False, inv_tr=False):
    t = [1 / N * i for i in range(N + 1)]  # discretisation "time"
    fig, ax = plt.subplots()
    plot_index = np.linspace(0, n, 5)
    plot_index = [math.floor(element) for element in plot_index]
    temp = []
    if acc_rej:
        for i in range(n):
            x, nr_it = acceptance_rejection(N)
            if i in plot_index:
                ax.plot(t, x)
            temp += [x]
    elif inv_tr:
        for i in range(n):
            x = inverse_transform(N)
            if i in plot_index:
                ax.plot(t, x)
            temp += [x]
    else:
        print('Choose an option! Not both!')
        return
    df_temp = pd.DataFrame(temp)
    x_mean = df_temp.mean(axis=0)
    x_var = df_temp.var(axis=0)
    ax.plot(t, x_mean, linestyle='dashed', label='Mean')
    ax.plot(t, x_var, linestyle='dotted', label='Variance')
    plt.show()


def exercise_1(T, N, n, std, plot=True, plot_real=True, mean_var=False):
    # Exercise 1a) and some plotting
    if not plot:
        plot_real = False
        mean_var = False
    if not plot_real:
        plot_wiener(T, N, n, std, plot_all=False)
    elif not mean_var:
        plot_wiener(T, N, n, std, mean=False, var=False)
    else:
        plot_wiener(T, N, n, std)

    # Exercise 1b)
    acceptance_rejection_test(N, plot=False)


def exercise_2(N, n, plot=True, acc_rej=True, inv_tr=False):
    if not acc_rej and not inv_tr:
        plot = False
    if plot:
        plot_realizations(N, n, acc_rej, inv_tr)
    else:
        print('What do you want then?')


# exercise_1(T, N, n, std, plot=True, plot_real=False, mean_var=True)
# exercise_2(N, n, acc_rej=True) # plot stuff




