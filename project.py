import numpy as np
import matplotlib.pyplot as plt
from Exercise1 import wiener_process


def C(sigma, t):
    mat = np.array([[2*t - np.sin(2*t), 1-np.cos(2*t)], [1-np.cos(2*t), 2*t + np.sin(2*t)]])
    c = (sigma**2/4)*mat
    return c


def Phi(t):
    phi = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return phi


def det_osc_1(x_0, y_0, T, N, sigma):
    h = T/N
    X_0 = np.array([x_0, y_0])
    X = [X_0]
    Phi_delta = Phi(h)
    for i in range(N):
        X_current = X[i]
        X_next = Phi_delta@X_current
        X += [X_next]

    return X


def stoch_osc_1(x_0, y_0, T, N, sigma):
    h = T/N
    X_0 = np.array([x_0, y_0])
    X = [X_0]
    Phi_delta = Phi(h)
    chol_C = np.linalg.cholesky(C(sigma, h))
    for i in range(N):
        r1 = np.random.normal()
        r2 = np.random.normal()
        r = np.array([r1, r2])
        xi_current = chol_C@r
        X_current = X[i]
        X_next = Phi_delta@X_current + xi_current
        X += [X_next]

    return X


def project_part_5_solver(stochastic=False):
    h = 1e-3
    T = 100
    N = int(T/h)
    t = np.linspace(0, T, N+1)
    sigma = 1
    x_0 = 1
    y_0 = 0

    if stochastic:
        X_stoch = stoch_osc_1(x_0, y_0, T, N, sigma)
        x = [X[0] for X in X_stoch]
        y = [X[1] for X in X_stoch]

    else:
        X_det = det_osc_1(x_0, y_0, T, N, sigma)
        x = [X[0] for X in X_det]
        y = [X[1] for X in X_det]

    return t, x, y


def project_part_5a():
    t, x_stoch, _ = project_part_5_solver(True)
    plt.rcParams['text.usetex'] = True
    plt.plot(t, x_stoch)
    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.title('Stochastic oscillator')
    plt.show()


def project_part_5b():
    t, x_stoch, y_stoch = project_part_5_solver(True)
    _, x_det, y_det = project_part_5_solver(False)
    plt.rcParams['text.usetex'] = True
    plt.plot(x_det, y_det, label='Deterministic oscillator')
    plt.plot(x_stoch, y_stoch, label='Stochastic oscillator')
    plt.xlabel('$x(t)$')
    plt.ylabel('$y(t)$')
    plt.legend()
    plt.title('Phase diagram of oscillators')
    plt.show()


def euler_maruyama_osc(x_0, y_0, sigma, T, N, stochastic=True):
    [t,W] = wiener_process(T, N)
    if not stochastic:
        W = np.zeros(len(t))
    h = T/N
    X_0 = np.array([x_0, y_0])
    X = [X_0]
    A = np.array([[0, 1], [-1, 0]])
    G = np.array([0, sigma])
    for i in range(N):
        f_i = A@X[i]
        new_X = X[i] + h*f_i + G*(W[i+1]-W[i])
        X += [new_X]
    return t, X


def calculate_energy(X):
    norm = np.linalg.norm(X, axis=1)
    E = 0.5*norm**2
    return E


def analytic_energy(t,sigma):
    E = [0.5*(1+sigma**2*s) for s in t]
    return E


def project_part_6_solver(stochastic=False):
    h = 5e-2
    T = 100
    N = int(T / h)
    sigma = 0.5
    x_0 = 1
    y_0 = 0

    t, X = euler_maruyama_osc(x_0, y_0, sigma, T, N, stochastic)
    x = [X[0] for X in X]
    y = [X[1] for X in X]

    return t, x, y


def project_part_6a():
    t, x, _ = project_part_6_solver(True)
    plt.rcParams['text.usetex'] = True
    plt.plot(t, x)
    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.title('Stochastic oscillator (Euler-Maruyama)')
    plt.show()


def project_part_6b():
    h = 5e-2
    T = 100
    N = int(T / h)
    sigma = 0.5
    x_0 = 1
    y_0 = 0
    nr_paths = 100
    E_sum = np.zeros(N+1)
    for n in range(nr_paths):
        _, X = euler_maruyama_osc(x_0, y_0, sigma, T, N, True)
        E_sum += calculate_energy(X)
    E_mean = (1/nr_paths)*E_sum
    t, _ = euler_maruyama_osc(x_0, y_0, sigma, T, N, True)
    E_real = analytic_energy(t, sigma)
    plt.rcParams['text.usetex'] = True
    plt.plot(t, E_mean, label='Euler-Maruyama')
    plt.plot(t, E_real, label='Real')
    plt.xlabel('$t$')
    plt.ylabel('$E(t)$')
    plt.title('Average energy over time')
    plt.legend()
    plt.show()


def project_part_6c():
    h = 5e-2
    T = 100
    N = int(T / h)
    sigma = 0.5
    x_0 = 1
    y_0 = 0
    nr_paths = 100
    E_sum_euler = np.zeros(N + 1)
    E_sum_expl = np.zeros(N+1)
    for n in range(nr_paths):
        _, X_euler = euler_maruyama_osc(x_0, y_0, sigma, T, N, True)
        X_expl = stoch_osc_1(x_0, y_0, T, N, sigma)
        E_sum_euler += calculate_energy(X_euler)
        E_sum_expl += calculate_energy(X_expl)

    E_mean_euler = (1 / nr_paths) * E_sum_euler
    E_mean_expl = (1 / nr_paths) * E_sum_expl
    t = np.linspace(0, T, N+1)
    E_real = analytic_energy(t, sigma)
    plt.rcParams['text.usetex'] = True
    plt.plot(t, E_mean_euler, label='Euler-Maruyama')
    plt.plot(t, E_mean_expl, label='Explicit', linestyle='dashed')
    plt.plot(t, E_real, label='Real')
    plt.xlabel('$t$')
    plt.ylabel('$E(t)$')
    plt.title('Average energy over time')
    plt.legend()
    plt.show()
    _, x_euler, y_euler = project_part_6_solver(True)
    plt.plot(x_euler, y_euler, label='Euler-Maruyama')
    _, x_expl, y_expl = project_part_5_solver(True)
    plt.plot(x_expl, y_expl, label='Explicit')
    plt.xlabel('$x(t)$')
    plt.ylabel('$y(t)$')
    plt.legend()
    plt.title('Phase diagram of explicit and Euler-maruyama')
    plt.show()

# project_part_5a()
# project_part_5b()
# project_part_6a()
# project_part_6b()
project_part_6c()






















