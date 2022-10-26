import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 1000  # no realizations
T = 1   # end of time interval
std = 1     # standard deviation of wiener process, needed?
N = 100     # points to simulate


def wiener_process(T, N, std=1, time=True):
    """
    Computes wiener process and time discretization vector
    :param float T: end time
    :param int N: number of points to simulate
    :param float std: standard deviation of wiener process
    :param bool time: choose wether or not to return time vector
    :return wiener process vector and (optionally) time vector:
    """
    h = T / N  # time step (equidistant)

    Z = [np.random.normal(0, std**2) for i in range(N + 1)]  # Vector of normal-distributed numbers

    W_0 = 0     # wiener process initial value

    W = [W_0]   # initiate vector of values for wiener process

    for i in range(len(Z) - 1):
        k = i + 1
        W += [W[k - 1] + np.sqrt(h) * Z[k]]     # fill out wiener process vector

    if time:
        t = [h * i for i in range(N + 1)]  # discretisation time
        return [t, W]
    else:
        return W


def plot_wiener(T, N, n=10, std=1, mean=True, var=True, plot_all=True):
    """
    Plots wiener process, as well as its mean and variance over time
    :param float T: end time
    :param int N: number of points to simulate
    :param int n: number of realizations
    :param float std: standard deviation of wiener process
    :param bool mean: wether or not to compute and plot mean
    :param bool var: wether or not to compute and plot variance
    :param bool plot_all: wether or not to plot realizations
    :return:
    """
    x = []
    wien_temp = wiener_process(T, N, std)
    t = wien_temp[0]    # get time discretization vector, so that we only have to do it once
    for j in range(n):
        x_temp = wiener_process(T, N, std, time=False)
        x += [x_temp]   # fill in n wiener process realizations in a list

    s_x = pd.DataFrame(x)   # collect the data in a pandas dataframe as they are very handy and easy to manage
    if plot_all:
        s_x.apply(lambda x: plt.plot(t, x), axis=1)     # plots all processes
    if var:
        wiener_var = s_x.var(0)    # computes variance of all realizations
        if (wiener_var.values < 0).any():
            print('Something wrong with variance. Negative values.')    # an initial test to make sure nothing is wrong
            # and that the variance is always non-negative
        else:
            plt.plot(t, wiener_var, linestyle='dotted', label='Variance')   # plot variance over time
    if mean:
        wiener_mean = s_x.mean(0)   # computes mean of all realizations
        plt.plot(t, wiener_mean, linestyle='dashed', label='Mean')  # plot mean over time
    if var or mean:
        plt.legend()    # plots legend so we know what's what
    plt.ylabel('W(t)')
    plt.xlabel('t [s]')
    plt.show()


def type_handler(x):
    """
    A type handler to make sure x is of the right type. Otherwise, make it into the right type, if possible.
    :param x:
    :return numpy array x, bool one_val:
    """
    if isinstance(x, (int, float)):
        x = np.array([x])
        one_val = True
    else:
        one_val = False
    return x, one_val


def check_range(x):
    """
    Make sure all values of a vector is in the desired range
    :param numpy array x:
    :return bool range_ok:
    """
    range_ok = True
    if (x < 0).any() or (x > 1).any():
        range_ok = False
    return range_ok


def pdf(x):
    """
    Probability density function.
    :param numpy array, float or int x:
    :return numpy array, float or int f:  the function evaluated at x
    """
    x, one_val = type_handler(x)    # check if type is okay, otherwise change it if possible
    range_ok = check_range(x)   # check if range is okay
    if not range_ok:
        print('x not in the right range')
        return
    f = [np.exp(y)/(np.exp(1)-1) for y in x]
    if one_val:
        f = f[0]    # we want to return it in the same form it came in. So if an array was given, we return an array.
        # But if an int or float was given we return a float.
    return f


def cdf(x):
    """
    Cumulative distribution function
    :param numpy array, float or int x:
    :return numpy array, float or int:  the function evaluated at x
    """
    return pdf(x)-pdf(0)


def inv_cdf(x):
    """
    Inverse cumulative distribution function
    :param numpy array, float or int x:
    :return numpy array, float or int G:  the function evaluated at x
    """
    x, one_val = type_handler(x)  # check if type is okay, otherwise change it if possible
    range_ok = check_range(x)  # check if range is okay
    if not range_ok:
        print('x not in the right range')
        return
    G = [np.log((np.exp(1)-1)*y + 1) for y in x]
    if one_val:
        G = G[0]    # we want to return it in the same form it came in. So if an array was given, we return an array.
        # But if an int or float was given we return a float.
    return G


def acceptance_rejection(N):
    """
    Acceptance-rejection method
    :param int N: number of samples
    :return list x, int nr_it: list of sampled points, number of iterations to attain the desired number of sampled points
    """
    c = pdf(1)
    i = 0   # number of samples done
    x = []  # initiate sample list
    nr_it = 0   # initiate number of iterations
    while i < N+1:  # check wether or not we have enough sample points
        [U, V] = np.random.uniform(0, 1, 2)     # draw two uniformly distributed numbers
        if U < pdf(V)/c:    # acceptance-rejection criteria
            x += [V]
            i += 1  # update the number of sample points we have
        nr_it += 1  # increase number of iterations variable after each iteration
    return x, nr_it


def acceptance_rejection_test(N, plot=True):
    """
    Tests the accpetance-rejection method by plotting histogram of samples together with probability density function.
    Optionally also returns acceptance rate
    :param int N: number of samples
    :param bool plot: wether or not to plot
    :return:
    """
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
    """
    Inverse transform method
    :param int N: number of samples
    :return list x, int nr_it: list of sampled points, number of iterations to attain the desired number of sampled points
    :param N:
    :return:
    """
    i = 0  # number of samples done
    x = []  # initiate sample list
    while i < N+1:  # check wether or not we have enough sample points
        U = np.random.uniform(0, 1)     # draw uniformly distributed number
        x += [inv_cdf(U)]
        i += 1
    return x


def inverse_transform_test(N):
    """
    Tests the inverse transform method by plotting histogram of samples together with probability density function.
    :param int N: number of samples
    :return:
    """
    X = inverse_transform(N)
    fig, ax = plt.subplots()
    temp = ax.hist(X, density=True)
    x = np.linspace(0, 1, 1000)
    f_x = pdf(x)
    ax.plot(x, f_x)
    plt.show()


def plot_realizations(N, n, acc_rej=False, inv_tr=False):
    """
    Plots realizations of the process with different methods
    :param int N: number of samples
    :param int n: number of realizations
    :param bool acc_rej: wether or not to use acceptance-rejection method
    :param bool inv_tr: wether or not to use inverse transform method
    :return:
    """
    t = [1 / N * i for i in range(N + 1)]  # discretisation "time"
    fig, ax = plt.subplots()
    plot_index = np.linspace(0, n, 5)  # use some kind of plotting "discretization" so we don't plot every single realization
    plot_index = [math.floor(element) for element in plot_index]    # make sure they are integers within the index range
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
    df_temp = pd.DataFrame(temp)    # store samples in a pandas dataframe
    x_mean = df_temp.mean(axis=0)   # compute mean
    x_var = df_temp.var(axis=0)     # compute variance
    ax.plot(t, x_mean, linestyle='dashed', label='Mean')
    ax.plot(t, x_var, linestyle='dotted', label='Variance')
    plt.legend()    # include legend so that we know what's what
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

