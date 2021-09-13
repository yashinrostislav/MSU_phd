from numpy import sin, cos, exp, linspace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

ALPHA = 1
BETA = 1
A = 1
B = 1


def f_1(T):
    return -(A * BETA * exp(ALPHA * T) * cos(BETA * T) +
             (B - A * ALPHA) * exp(-ALPHA * T) * sin(BETA * T))


def f_2(T):
    return -(A * BETA * exp(ALPHA * T) * cos(BETA * T) +
             BETA ** 2 * A * exp(ALPHA * T) * (-sin(BETA * T)) +
             (B - A * ALPHA) * (ALPHA * exp(ALPHA * T) * sin(BETA * T) +
                                BETA * exp(-ALPHA * T) * cos(BETA * T)))


def func_P(T):
    coef_1 = 1 / (4 * ALPHA ** 2 + 4 * BETA ** 2)
    return (-exp(-ALPHA ** T) / (4 * ALPHA) * sin(BETA * T) +
            coef_1 * (-BETA * exp(ALPHA * T) * cos(BETA * T) - ALPHA * exp(ALPHA * T) * sin(BETA)) +
            coef_1 * exp(-2 * ALPHA * T) *
            (-exp(ALPHA * T) * cos(BETA * T) * (-ALPHA * sin(2 * BETA * T) - BETA * cos(2 * BETA * T)) -
             exp(ALPHA * T) * sin(BETA * T) * (ALPHA * cos(2 * BETA * T) + BETA * sin(2 * BETA * T))))


def func_Q(T):
    coef_1 = 1 / (4 * ALPHA ** 2 + 4 * BETA ** 2)
    coef_2 = 1 / (4 * ALPHA)
    return (coef_2 * (exp(-ALPHA * T) * cos(BETA * T) -
                      exp(ALPHA * T) * cos(BETA * T)) +
            coef_1 * (BETA * exp(ALPHA * T) * sin(BETA * T) -
                      ALPHA * exp(ALPHA * T) * cos(BETA * T)) +
            coef_2 * exp(-2 * ALPHA * T) *
            (exp(ALPHA * T) * sin(BETA * T) * (-ALPHA * sin(BETA * T) -
                                               BETA * cos(2 * BETA * T)) -
             exp(ALPHA * T) * cos(BETA * T) * (ALPHA * cos(2 * BETA * T) +
                                               BETA * sin(2 * BETA * T))))


def func_A(T):
    coef_1 = 1 / (4 * ALPHA ** 2 + 4 * BETA ** 2)
    return (ALPHA * func_P(T) + BETA * exp(ALPHA * T) / 2 *
            (cos(BETA * T) * 2 * coef_1 * (ALPHA + (-ALPHA * cos(2 * BETA * T) +
                                                    BETA * sin(2 * BETA * T)) * exp(-2 * ALPHA * T)) -
             sin(BETA * T) * 2 * coef_1 * (BETA + (ALPHA * sin(2 * BETA * T) +
                                                   BETA * cos(2 * BETA * T)) * exp(-2 * ALPHA * T)) +
             1 / (2 * ALPHA) * cos(BETA * T) * (exp(-2 * ALPHA * T) - 1)))


def func_B(T):
    coef_1 = 1 / (4 * ALPHA ** 2 + 4 * BETA ** 2)
    return (ALPHA * func_Q(T) + BETA * exp(ALPHA * T) / 2 *
            (-cos(BETA * T) * 2 * coef_1 * (ALPHA * sin(2 * BETA * T) +
                                            BETA * cos(2 * BETA * T) * exp(-2 * ALPHA * T) +
                                            BETA) -
             sin(BETA * T) * 2 * coef_1 * (-ALPHA * cos(2 * BETA * T) +
                                           BETA * sin(2 * BETA * T) * exp(-2 * ALPHA * T) +
                                           ALPHA) -
             1 / (2 * ALPHA) * sin(BETA * T) * (exp(-2 * ALPHA * T) - 1)))


def c_1(T):
    return (-func_B(T) * f_1(T) + f_2(T) * func_Q(T) /
            func_A(T) * func_Q(T) - func_B(T) * func_P(T))


def c_2(T):
    return (func_A(T) * f_1(T) - f_2(T) * func_P(T) /
            func_A(T) * func_Q(T) - func_B(T) * func_P(T))


def g(t, T):
    return (c_1(T) * exp(-ALPHA * t) * cos(BETA * t) +
            c_2(T) * exp(-ALPHA * t) * sin(BETA * t))


def plot_2d_fig(t_time=linspace(1, 5, 1001),
                T=3,
                save_pic=True):
    plt.plot(t_time, g(t_time, T))
    if save_pic:
        plt.savefig('func_2d_g.png')
    plt.show()


def plot_3d_fig(t_time=np.linspace(1, 5, 101),
                T_period=np.linspace(6, 8, 101),
                save_pic=True):
    B, D = np.meshgrid(t_time, T_period)
    g_func = g(B, D)

    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    # ax.zaxis.set_major_locator(LinearLocator(10))

    surf = ax.plot_surface(B, D, g_func, cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=20)
    plt.xlabel('t')
    plt.ylabel('T')
    ax.view_init(30, 120)
    if save_pic:
        plt.savefig('func_3d_g.png')
    plt.show()


if __name__ == '__main__':
    plot_2d_fig()
    plot_3d_fig()
