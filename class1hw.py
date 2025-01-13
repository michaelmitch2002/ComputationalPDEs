import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy

u0 = 1
tf = 6
h = 0.25

def fe_central(tf, h):
    def fe(unow, uprev):
        unext = -2*h*unow + uprev
        return unext
    Nt = int(tf / h) + 1
    u = np.zeros(Nt)
    t = np.linspace(0, tf, Nt)
    u1 = np.exp(-h)
    u[0] = u0
    u[1] = u1
    for i in range(1, Nt - 1):
        u[i + 1] = fe(u[i], u[i - 1])
    return t, u


t, u = fe_central(tf, h)
ureal = np.exp(-t)
plt.plot(t, u)
plt.plot(t, ureal)
plt.show()

