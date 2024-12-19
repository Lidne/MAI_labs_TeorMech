import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

a = 1.0  # расстояние между центрами пружин D и E
ab_l = 2.0  # длина стержня AB
g = 9.81  # ускорение свободного падения
m2 = 1  # масса груза
slider_height = 1.4
slider_y = -0.5  # высота ползуна
# точки
point_D = [a, 0]
point_O = [0, 0]
point_E = [-a, 0]
wall_D = [a + 0.05, 0]
wall_E = [-a - 0.05, 0]
fi0 = np.pi / 4  # начальный угол от вертикали

t_max = 5.0
dt = 0.08
t_values = np.arange(0, t_max, dt)

fig = plt.figure()
gs = GridSpec(5, 1, figure=fig)

ax = fig.add_subplot(gs[:5, :])
(line,) = ax.plot([], [], "c-")  # стержень AB
(ball,) = ax.plot([], [], "bo", markersize=10)  # груз B
(slider,) = ax.plot([], [], "go", markersize=15)  # ползун A
(point_d,) = ax.plot([], [], "co", markersize=3)  # точка D
(point_o,) = ax.plot([], [], "mo", markersize=3)  # точка O
(point_e,) = ax.plot([], [], "co", markersize=3)  # точка E
(wall_d,) = ax.plot([], [], "bo", markersize=8)  # стенка D
(wall_e,) = ax.plot([], [], "bo", markersize=8)  # стенка E
(spring_DA,) = ax.plot([], [], "r", lw=2)  # пружина DA
(spring_EA,) = ax.plot([], [], "r", lw=2)  # пружина EA
ax.set_xlim(-10, 10)
ax.set_ylim(-3, 1)
ax.axvline(0, linestyle="--", color="k")  # пунктиры
ax.axhline(point_D[1], linestyle="--", color="k")

point_d.set_data([point_D[0]], [point_D[1]])
point_o.set_data([point_O[0]], [point_O[1]])
point_e.set_data([point_E[0]], [point_E[1]])

wall_d.set_data([wall_D[0]], [wall_D[1]])
wall_e.set_data([wall_E[0]], [wall_E[1]])
# подписи точек
ax.text(point_D[0], point_D[1], "D", ha="right", va="bottom")
ax.text(point_O[0], point_O[1], "O", ha="right", va="bottom")
ax.text(point_E[0], point_E[1], "E", ha="right", va="bottom")

# буквы A и B
text_A = ax.text(0, slider_y, "A", ha="right", va="bottom")
text_B = ax.text(ab_l * np.sin(fi0), slider_y - ab_l * np.cos(fi0), "B", ha="right", va="bottom")


# задаем пружины
spring_DA.set_data(point_D, [0, slider_y - 0.1])
spring_EA.set_data(point_E, [0, slider_y - 0.1])


def animate(i):
    global slider_y
    t = i * dt

    # меняем положение ползуна
    slider_y = -ab_l * np.cos(fi0 * np.cos(np.sqrt(g / ab_l) * t)) + slider_height

    # уравнение маятника
    fi = fi0 * np.cos(np.sqrt(g / ab_l) * t)

    x_values = [0, ab_l * np.sin(fi)]
    y_values = [slider_y, slider_y - ab_l * np.cos(fi)]
    line.set_data(x_values, y_values)

    # положение груза B
    x_ball = ab_l * np.sin(fi)
    y_ball = slider_y - ab_l * np.cos(fi)
    ball.set_data([x_ball], [y_ball])

    # обновляем пружины
    spring_DA.set_data(point_D, [0, slider_y - 0.1])
    spring_EA.set_data(point_E, [0, slider_y - 0.1])

    # положение ползуна
    slider.set_data([0], [slider_y - 0.1])

    # обновляем координаты букв
    text_A.set_position((0, slider_y))
    text_B.set_position((x_ball, y_ball))

    return line, ball, spring_EA, spring_DA, slider, point_d, point_o, point_e, text_A, text_B, wall_d, wall_e


ani = FuncAnimation(fig, animate, frames=len(t_values), blit=True, interval=100)
plt.show()
