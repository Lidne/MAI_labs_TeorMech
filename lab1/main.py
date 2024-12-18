import sympy as sp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# def Rot2D(x, y, alpha):
#     RX =


T = np.linspace(0, 10, 1000)
t = sp.Symbol("t")

R = 1
r = sp.cos(6 * t)
fi = t + 0.2 * sp.cos(3 * t)

x = R * r * sp.cos(fi)
y = R * r * sp.sin(fi)

x_diff = sp.diff(x, t)
y_diff = sp.diff(y, t)
x_diff2 = sp.diff(x_diff, t)
y_diff2 = sp.diff(y_diff, t)

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
VVX = np.zeros_like(T)
VVY = np.zeros_like(T)

for i in range(len(T)):
    X[i] = float(x.subs(t, T[i]))
    Y[i] = float(y.subs(t, T[i]))
    VX[i] = float(x_diff.subs(t, T[i]))
    VY[i] = float(y_diff.subs(t, T[i]))
    VVX[i] = float(x_diff2.subs(t, T[i]))
    VVY[i] = float(y_diff2.subs(t, T[i]))


fig, ax = plt.subplots()
ax.axis("equal")
ax.set_xlim([min(X) - 0.5, max(X) + 0.5])
ax.set_ylim([min(Y) - 0.5, max(Y) + 0.5])

ArrowX = np.array([-0.2 * R, 0, -0.2 * R])
ArrowY = np.array([0.1 * R, 0, -0.1 * R])

ax.plot(ArrowX, ArrowY, "b-")


(point,) = ax.plot([], [], "go", markersize=10)
ax.plot(X, Y, "r-", lw=1)
(VLine,) = ax.plot([], [], "b-", lw=1)


def init():
    point.set_data([], [])
    return (point,)


def update(frame):
    point.set_data([X[frame]], [Y[frame]])
    VLine.set_data([X[frame], X[frame] + VX[frame] / 3], [Y[frame], Y[frame] + VY[frame] / 3])
    ArrowY
    return point, VLine


ani = animation.FuncAnimation(fig, update, frames=len(T), init_func=init, interval=20, blit=True)

plt.show()
