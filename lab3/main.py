import sympy as sp

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# from matplotlib.patches import Rectangle

# from matplotlib.patches import Circle

# import math

from scipy.integrate import odeint


def formY(y, t, fv, fw):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, fv(y1, y2, y3, y4), fw(y1, y2, y3, y4)]
    return dydt


Frames = 500
t = sp.Symbol("t")  # символ времени
x = sp.Function("x")(t)  # положение прямоугольника
fi = sp.Function("fi")(t)  # угол маятника
v = sp.Function("v")(t)  # скорость прямоугольника
w = sp.Function("w")(t)  # угловая скорость маятника

width = 1  # ширина ползуна
length = 2  # длина ползуна
circle_radius = 0.2  # радиус шара
a = 1  # (DO и OE)
m1 = 1  # масса ползуна
m2 = 1  # масса шара
g = 9.8
pendulum_l = 2  # длина маятника
k = 5  # коэф. жесткости
y0 = [-5, sp.rad(45), 0, 0]  # x(0), fi(0), v(0), w(0)

# кинетическая энергия ползуна
Ekin1 = (m1 * v * v) / 2
# квадратичная скорость центра масс
Vsquared = v * v + w * w * pendulum_l * pendulum_l - 2 * v * w * pendulum_l * sp.sin(fi)
# кинетическая энергия маятника
pend_kin = m2 * Vsquared / 2
# кин. энергия системы
system_kin = Ekin1 + pend_kin
# потенциальная энергия
spring_dx = sp.sqrt(a * a + x * x) - a  # изменение длины пружины
springs_pot = k * spring_dx * spring_dx
system_pot = -m1 * g * x - m2 * g * (x + pendulum_l * sp.cos(fi)) + springs_pot
# generalized forces
Qx = -sp.diff(system_pot, x)
Qfi = -sp.diff(system_pot, fi)
# считаем лагранжиану
Lagr = system_kin - system_pot
ur1 = sp.diff(sp.diff(Lagr, v), t) - sp.diff(Lagr, x)
ur2 = sp.diff(sp.diff(Lagr, w), t) - sp.diff(Lagr, fi)

# метод крамера
a11 = ur1.coeff(sp.diff(v, t), 1)
a12 = ur1.coeff(sp.diff(w, t), 1)
a21 = ur2.coeff(sp.diff(v, t), 1)
a22 = ur2.coeff(sp.diff(w, t), 1)
b1 = -(ur1.coeff(sp.diff(v, t), 0)).coeff(sp.diff(w, t), 0).subs([(sp.diff(x, t), v), (sp.diff(fi, t), w)])
b2 = -(ur2.coeff(sp.diff(v, t), 0)).coeff(sp.diff(w, t), 0).subs([(sp.diff(x, t), v), (sp.diff(fi, t), w)])
detA = a11 * a22 - a12 * a21
detA1 = b1 * a22 - b2 * a21
detA2 = a11 * b2 - b1 * a21
dvdt = detA1 / detA
dwdt = detA2 / detA
# массив кадров
T = np.linspace(0, 50, Frames)
# переводим в лямдба функции
fv = sp.lambdify([x, fi, v, w], dvdt, "numpy")
fw = sp.lambdify([x, fi, v, w], dwdt, "numpy")
sol = odeint(formY, y0, T, args=(fv, fw))
# sol[:,0] - x
# sol[:,1] - fi
# sol[:,2] - v (dx/dt)
# sol[:,3] - w (dfi/dt)
ax = sp.lambdify(x, 0)
ay = sp.lambdify(x, x)
AX = ax(sol[:, 0])
AY = -ay(sol[:, 0])

bx = sp.lambdify(fi, pendulum_l * sp.sin(fi))
by = sp.lambdify([x, fi], +pendulum_l * sp.cos(fi) + x)
BX = bx(sol[:, 1])
BY = -by(sol[:, 0], sol[:, 1])

# Расчет R_a
phi_prime = sp.diff(fi, t)
phi_double_prime = sp.diff(phi_prime, t)
Ra_expr = m2 * pendulum_l * (phi_double_prime * sp.cos(fi) - phi_prime**2 * sp.sin(fi))
Ra_func = sp.lambdify([fi, phi_prime, phi_double_prime], Ra_expr, "numpy")

# Вычисление значений R_a
phi_prime_vals = np.gradient(sol[:, 1], T)
phi_double_prime_vals = np.gradient(phi_prime_vals, T)
Ra_vals = Ra_func(sol[:, 1], phi_prime_vals, phi_double_prime_vals)

fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax0.axis("equal")

L1X = [-width / 2, -width / 2]
L2X = [width / 2, width / 2]
LY = [min(AY) - length, max(AY) + length]

ax0.plot(L1X, LY, color="grey")  # левая стенка
ax0.plot(L2X, LY, color="grey")  # правая стенка
(sl,) = ax0.plot([-a, -length / 2], [0, AY[0] + width / 2], color="brown")  # левая пружина
(sr,) = ax0.plot([a, length / 2], [0, AY[0] + width / 2], color="brown")  # правая пружина
ax0.plot(-a, 0, marker=".", color="black")  # левое крепление пружины
ax0.plot(a, 0, marker=".", color="black")  # правое крепление пружины
rect = plt.Rectangle((-width / 2, AY[0]), width, length, color="black")  # ползун
circ = plt.Circle((BX[0], BY[0]), circle_radius, color="grey")  # шарик
(R_vector,) = ax0.plot([0, BX[0]], [0, BY[0]], color="grey")


# графики
ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(T, sol[:, 0])
ax2.set_xlabel("t")
ax2.set_ylabel("x")
ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, sol[:, 1])
ax3.set_xlabel("t")
ax3.set_ylabel("fi")
ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(T, Ra_vals)
ax4.set_xlabel("t")
ax4.set_ylabel("Ra")
plt.subplots_adjust(wspace=0.3, hspace=0.7)


def init():
    rect.set_y(-length / 2)
    ax0.add_patch(rect)
    circ.center = (0, 0)
    ax0.add_patch(circ)
    return rect, circ


def anima(i):
    rect.set_y(AY[i] - length / 2)
    sl.set_data([-a, -width / 2], [0, AY[i]])
    sr.set_data([a, width / 2], [0, AY[i]])
    R_vector.set_data([0, BX[i]], [AY[i], BY[i]])
    circ.center = (BX[i], BY[i])
    return (sl, sr, rect, R_vector, circ)


anim = FuncAnimation(
    fig,
    anima,
    init_func=init,
    frames=Frames,
    interval=10,
    blit=False,
    repeat=True,
    repeat_delay=0,
)
plt.show()
