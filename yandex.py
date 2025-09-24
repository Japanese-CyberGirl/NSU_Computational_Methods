import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- параметры маятника ---
g = 9.81      # ускорение свободного падения (м/с^2)
L = 1.0       # длина нити (м)
theta0 = 0.3  # начальное отклонение (рад)
omega0 = 0.0  # начальная угловая скорость (рад/с)
dt = 0.02     # шаг по времени
T = 20        # общее время моделирования (с)

# --- функция правых частей ---
def d2theta_dt2(theta, omega):
    return -(g / L) * np.sin(theta)

# --- метод Эйлера для численного решения ---
t = np.arange(0, T, dt)
theta = np.zeros_like(t)
omega = np.zeros_like(t)

theta[0] = theta0
omega[0] = omega0

for i in range(1, len(t)):
    omega[i] = omega[i-1] + d2theta_dt2(theta[i-1], omega[i-1]) * dt
    theta[i] = theta[i-1] + omega[i] * dt

# --- визуализация ---
x = L * np.sin(theta)
y = -L * np.cos(theta)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(-1.2*L, 1.2*L)
ax.set_ylim(-1.2*L, 0.2*L)
ax.set_aspect('equal')
ax.set_title("Колебания маятника")

line, = ax.plot([], [], 'o-', lw=2)

def update(i):
    line.set_data([0, x[i]], [0, y[i]])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
plt.show()
