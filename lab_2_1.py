import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def original_function(c):
    m = 68.1
    t = 10
    g = 9.8
    return (g * m / c) * (1 - np.exp(- (c / m) * t))

answer = 40

c_values = []
v_values = []

for i in range(-10, 30):
    if i == 0:
        continue
    c_values.append(i)
    v_values.append(original_function(i))

fig_1 = plt.figure(figsize=(12, 8))
plt.plot(c_values, v_values, 'b-', linewidth=2, label='Скорость V')
plt.xlabel('Коэффициент лобового сопротивления')
plt.axhline(y=answer, color='r', linestyle='--', linewidth=2, label=f"Ожидаемый ответ = {answer}")
plt.ylabel('Значение скорости')
plt.title('Изменение коэффициента лобового сопротивления к скорости')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

def fixed_function(c):
    m = 68.1
    t = 10
    g = 9.8
    answer = 40
    if c == 0:
        return np.nan
    return (g * m / c) * (1 - np.exp(- (c / m) * t)) - 40

c_values.clear()
v_values.clear()

for i in range(-10, 30):
    if i == 0:
        continue
    c_values.append(i)
    v_values.append(fixed_function(i))

fig_2 = plt.figure(figsize=(12, 8))
plt.plot(c_values, v_values, 'b-', linewidth=2, label='Разница искомой скорости и фактической')
plt.xlabel('Коэффициент лобового сопротивления')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label=f"Ожидаемая разница = 0")
plt.ylabel('Значение разности скоростей')
plt.title('Изменение коэффициента лобового сопротивления к скорости')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

global epsilon
epsilon = 1e-6

def bisection(a, b):
    iterations = 0
    values = []
    while (b - a) > epsilon:
        iterations += 1
        c = (a + b) / 2
        values.append(c)
        if fixed_function(a) * fixed_function(c) < 0:
            b = c
        elif fixed_function(a) * fixed_function(c) > 0:
            a = c
        else:
            return c, iterations, values
    x = (a + b) / 2
    return x, iterations, values

bisection_root, bisection_iterations, bisection_values = bisection(1, 20)
bisection_iterations_array = [int(i) for i in range(bisection_iterations)]


barier_d = 0

for i in range(bisection_iterations):
    if abs(bisection_values[i] - bisection_root) < 0.01:
        barier_d = i
        break

fig_3 = plt.figure(figsize=(12, 8))
plt.plot(bisection_iterations_array, bisection_values, 'b-', linewidth=2, label='Значение бисекции')
plt.xlabel('Итерации')
plt.axvline(barier_d)
plt.axhline(y=bisection_root, color='r', linestyle='--', linewidth=2, label=f"Найденный корень = {bisection_root:.4f}")
plt.ylabel('Значение c')
plt.title('Приближение методом дихотомии')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе бисекции достигается на итерации под номером = {barier_d}")


def regula_falsi(a , b):
    iterations = 0
    values = []
    f_a = fixed_function(a)
    f_b = fixed_function(b)
    c = a

    while (b - a > epsilon):
        iterations += 1
        c = a - f_a * (b - a) / (f_b - f_a)
        values.append(c)
        f_c = fixed_function(c)

        if abs(f_c) < epsilon:
            return c , iterations , values
        if (f_a * f_c < 0):
            b = c
            f_b = f_c
        else:
            a = c
            f_a = f_c

    return c , iterations , values

regula_root, regula_iterations, regula_values = regula_falsi(1, 20)
regula_iterations_array = [int(i) for i in range(regula_iterations)]

barier_r = 0
for i in range(regula_iterations):
    if abs(regula_values[i] - regula_root) < 0.001:
        barier_r = i
        break

fig_4 = plt.figure(figsize=(12, 8))
plt.plot(regula_iterations_array, regula_values, 'g-', linewidth=2, label='Значение regula falsi')
plt.xlabel('Итерации')
plt.axvline(barier_r, color='orange', linestyle='--', linewidth=2, label=f'ε < 0.001 на итерации {barier_r}')
plt.axhline(y=regula_root, color='r', linestyle='--', linewidth=2, label=f"Найденный корень = {regula_root:.4f}")
plt.ylabel('Значение c')
plt.title('Приближение методом regula falsi')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе regula falsi достигается на итерации под номером = {barier_r}")


def mod_regula_falsi(a, b):
    iterations = 0
    values = []
    f_a = fixed_function(a)
    f_b = fixed_function(b)
    c = a
    i_a = 0
    i_b = 0

    while (b - a) > epsilon:
        iterations += 1
        c = b - f_b * (b - a) / (f_b - f_a)
        values.append(c)
        f_c = fixed_function(c)

        if abs(f_c) < epsilon:
            return c, iterations, values

        temp = f_a * f_c
        if temp < 0:
            b = c
            f_b = f_c
            i_b = 0
            i_a += 1
            if i_a >= 2:
                f_a /= 2
        elif temp > 0:
            a = c
            f_a = f_c
            i_a = 0
            i_b += 1
            if i_b >= 2:
                f_b /= 2
        else:
            return c, iterations, values

    return c, iterations, values


mod_root, mod_iterations, mod_values = mod_regula_falsi(1, 20)
mod_iterations_array = [int(i) for i in range(mod_iterations)]

barier_m = 0
for i in range(mod_iterations):
    if abs(mod_values[i] - mod_root) < 0.01:
        barier_m = i
        break

fig_5 = plt.figure(figsize=(12, 8))
plt.plot(mod_iterations_array, mod_values, 'm-', linewidth=2, label='Значение mod regula falsi')
plt.xlabel('Итерации')
plt.axvline(barier_m, color='orange', linestyle='--', linewidth=2, label=f'ε < 0.001 на итерации {barier_m}')
plt.axhline(y=mod_root, color='r', linestyle='--', linewidth=2, label=f"Найденный корень = {mod_root:.4f}")
plt.ylabel('Значение c')
plt.title('Приближение методом модифицированного regula falsi')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе mod regula falsi достигается на итерации под номером = {barier_m}")
