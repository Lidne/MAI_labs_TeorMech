import sympy as sp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import math


def Rot2D(X, Y, Alpha):
    """Функция для поворота"""
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


T = np.linspace(0, 10, 1000)
t = sp.Symbol("t")

# условние
R = 1
r = 2 + sp.cos(6 * t)
fi = 7 * t + 1.2 * sp.cos(6 * t)

# перевод из полярных в декартовы координаты
x = R * r * sp.cos(fi)
y = R * r * sp.sin(fi)

# вычисление скорости и ускорения
x_diff = sp.diff(x, t)
y_diff = sp.diff(y, t)
x_diff2 = sp.diff(x_diff, t)
y_diff2 = sp.diff(y_diff, t)

# массивы с значениями в каждый момент времени t
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
VVX = np.zeros_like(T)
VVY = np.zeros_like(T)

# расчет значений
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

SPD_V_REDUCE_KOEF = 3
ACC_V_REDUCE_KOEF = 30

ArrowX = np.array([-0.2 * R, 0, -0.2 * R])
ArrowY = np.array([0.1 * R, 0, -0.1 * R])

# стрелки
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))

# стрелка вектора скорости
(VArrow,) = ax.plot(RArrowX + X[0] + VX[0] / 3, RArrowY + Y[0] + VY[0] / 3, "b")

# стрелка вектора ускорения
(VVArrow,) = ax.plot(RArrowX + X[0] + VVX[0] / 3, RArrowY + Y[0] + VVY[0] / 3, "g")

# стрелка радиус-вектора
(RadArrow,) = ax.plot(RArrowX + X[0], RArrowY + Y[0], "r")


(point,) = ax.plot([], [], "go", markersize=10)
ax.plot(X, Y, "r-", lw=1)

# линия вектора скорости
(VLine,) = ax.plot([], [], "b-", lw=1)

# линия вектора ускорения
(VVLine,) = ax.plot([], [], "g-", lw=1)

# линия радиус-вектора
(RadLine,) = ax.plot([], [], "r-", lw=1)


def init():
    point.set_data([], [])
    return (point,)


def update(frame):
    """Функция для отрисовки нового кадра"""
    point.set_data([X[frame]], [Y[frame]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[frame], VX[frame]))
    VArrow.set_data(
        RArrowX + X[frame] + VX[frame] / SPD_V_REDUCE_KOEF, RArrowY + Y[frame] + VY[frame] / SPD_V_REDUCE_KOEF
    )
    VLine.set_data(
        [X[frame], X[frame] + VX[frame] / SPD_V_REDUCE_KOEF], [Y[frame], Y[frame] + VY[frame] / SPD_V_REDUCE_KOEF]
    )

    VRArrowX, VRArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VVY[frame], VVX[frame]))
    VVArrow.set_data(
        VRArrowX + X[frame] + VVX[frame] / ACC_V_REDUCE_KOEF, VRArrowY + Y[frame] + VVY[frame] / ACC_V_REDUCE_KOEF
    )
    VVLine.set_data(
        [X[frame], X[frame] + VVX[frame] / ACC_V_REDUCE_KOEF], [Y[frame], Y[frame] + VVY[frame] / ACC_V_REDUCE_KOEF]
    )

    RadArrowX, RadArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[frame], X[frame]))
    RadArrow.set_data(RadArrowX + X[frame], RadArrowY + Y[frame])
    RadLine.set_data([0, X[frame]], [0, Y[frame]])
    return point, VLine, VArrow, VVLine, VVArrow, RadArrow, RadLine


# запуск анимации
ani = animation.FuncAnimation(fig, update, frames=len(T), init_func=init, interval=40, blit=True)

plt.show()
