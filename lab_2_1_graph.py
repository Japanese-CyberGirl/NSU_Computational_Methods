import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def fixed_function(c):
    m = 68.1
    t = 10
    g = 9.8
    answer = 40
    if c == 0:
        return np.nan
    return (g * m / c) * (1 - np.exp(- (c / m) * t)) - 40


global epsilon
epsilon = 1e-16



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



start_time = time.perf_counter()
bisection_root, bisection_iterations, bisection_values = bisection(1, 20)
bisection_time = time.perf_counter() - start_time

print(f"Время выполнения алгоритма метода бисекций = {bisection_time:.8f} сек")

bisection_iterations_array = list(range(bisection_iterations))
bisection_function_root = abs(fixed_function(bisection_root))
bisection_function_values = [abs(fixed_function(i)) for i in bisection_values]

print(f"Количество итераций в методе бисекции = {bisection_iterations}")

barier_b = 0
for i in range(bisection_iterations):
    if bisection_function_values[i] < 1e-3:
        barier_b = i
        break

print("barier_b =", barier_b)

fig_3 = plt.figure(figsize=(12, 8))
plt.semilogy(bisection_iterations_array, bisection_function_values, 'b-', linewidth=2, label='f(c) бисекции')
plt.axvline(barier_b, color='orange', linestyle='--', linewidth=2, label=f"barier_b = {barier_b}")
plt.xlabel('Итерации')
plt.axhline(y=bisection_function_root, color='r', linestyle='--', linewidth=2)
plt.ylabel('Значение f(c)')
plt.title('Приближение методом бисекции')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()





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



start_time = time.perf_counter()
regula_root, regula_iterations, regula_values = regula_falsi(1, 20)
regula_time = time.perf_counter() - start_time

print(f"Время выполнения алгоритма метода regula falsi = {regula_time:.8f} сек")

regula_iterations_array = list(range(regula_iterations))
regula_function_root = abs(fixed_function(regula_root))
regula_function_values = [abs(fixed_function(i)) for i in regula_values]

print(f"Количество итераций в методе regula falsi = {regula_iterations}")

barier_r = 0
for i in range(regula_iterations):
    if regula_function_values[i] < 1e-3:
        barier_r = i
        break

print("barier_r =", barier_r)

fig_4 = plt.figure(figsize=(12, 8))
plt.semilogy(regula_iterations_array, regula_function_values, 'b-', linewidth=2, label='f(c) regula falsi')
plt.axvline(barier_r, color='orange', linestyle='--', linewidth=2, label=f"barier_r = {barier_r}")
plt.xlabel('Итерации')
plt.ylabel('Значение f(c)')
plt.title('Приближение методом regula falsi')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()




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



start_time = time.perf_counter()
mod_root, mod_iterations, mod_values = mod_regula_falsi(1, 20)
mod_time = time.perf_counter() - start_time

print(f"Время выполнения алгоритма метода mod. regula falsi = {mod_time:.8f} сек")

mod_iterations_array = list(range(mod_iterations))
mod_function_root = abs(fixed_function(mod_root))
mod_function_values = [abs(fixed_function(i)) for i in mod_values]

print(f"Количество итераций в методе mod. regula falsi = {mod_iterations}")

barier_m = 0
for i in range(mod_iterations):
    if mod_function_values[i] < 1e-3:
        barier_m = i
        break

print("barier_m =", barier_m)

fig_5 = plt.figure(figsize=(12, 8))
plt.semilogy(mod_iterations_array, mod_function_values, 'b-', linewidth=2, label='f(c) mod regula falsi')
plt.axvline(barier_m, color='orange', linestyle='--', linewidth=2, label=f"barier_m = {barier_m}")
plt.xlabel('Итерации')
plt.ylabel('Значение f(c)')
plt.title('Приближение методом mod regula falsi')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()





fig_all = plt.figure(figsize=(14, 9))

plt.semilogy(bisection_iterations_array, bisection_function_values, linewidth=2, label='Бисекция')
plt.semilogy(regula_iterations_array, regula_function_values, linewidth=2, label='Regula Falsi')
plt.semilogy(mod_iterations_array, mod_function_values, linewidth=2, label='Mod. Regula Falsi')

plt.axvline(barier_b, color='blue', linestyle='--', linewidth=2, label=f'barier_b = {barier_b}')
plt.axvline(barier_r, color='green', linestyle='--', linewidth=2, label=f'barier_r = {barier_r}')
plt.axvline(barier_m, color='purple', linestyle='--', linewidth=2, label=f'barier_m = {barier_m}')

plt.axhline(1e-3, color='orange', linestyle=':', linewidth=2)

plt.xlabel('Итерации')
plt.ylabel('|f(c)|')
plt.title('Сравнение методов по итерациям')
plt.grid(True, alpha=0.3, which='both')
plt.legend()
plt.show()




bisection_times = []
start = time.perf_counter()
for _ in bisection_values:
    bisection_times.append(time.perf_counter() - start)


fig_bt = plt.figure(figsize=(12, 8))
plt.semilogy(bisection_times, bisection_function_values, 'b-', linewidth=2, label='Бисекция f(c)')
plt.axvline(bisection_times[barier_b], color='orange', linestyle='--', linewidth=2, label=f"barier_b")
plt.xlabel('Время (сек)')
plt.ylabel('|f(c)|')
plt.title('Сходимость метода бисекции во времени')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()



regula_times = []
start = time.perf_counter()
for _ in regula_values:
    regula_times.append(time.perf_counter() - start)


fig_rt = plt.figure(figsize=(12, 8))
plt.semilogy(regula_times, regula_function_values, 'g-', linewidth=2, label='Regula Falsi f(c)')
plt.axvline(regula_times[barier_r], color='orange', linestyle='--', linewidth=2, label=f"barier_r")
plt.xlabel('Время (сек)')
plt.ylabel('|f(c)|')
plt.title('Сходимость метода regula falsi во времени')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()



mod_times = []
start = time.perf_counter()
for _ in mod_values:
    mod_times.append(time.perf_counter() - start)


fig_mt = plt.figure(figsize=(12, 8))
plt.semilogy(mod_times, mod_function_values, 'm-', linewidth=2, label='Mod Regula Falsi f(c)')
plt.axvline(mod_times[barier_m], color='orange', linestyle='--', linewidth=2, label=f"barier_m")
plt.xlabel('Время (сек)')
plt.ylabel('|f(c)|')
plt.title('Сходимость метода mod regula falsi во времени')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()





fig_all_time = plt.figure(figsize=(14, 9))

plt.semilogy(bisection_times, bisection_function_values, linewidth=2, label='Бисекция')
plt.semilogy(regula_times, regula_function_values, linewidth=2, label='Regula Falsi')
plt.semilogy(mod_times, mod_function_values, linewidth=2, label='Mod. Regula Falsi')

plt.axvline(bisection_times[barier_b], color='blue', linestyle='--', linewidth=2)
plt.axvline(regula_times[barier_r], color='green', linestyle='--', linewidth=2)
plt.axvline(mod_times[barier_m], color='purple', linestyle='--', linewidth=2)

plt.axhline(1e-3, color='orange', linestyle=':', linewidth=2)

plt.xlabel('Время (сек)')
plt.ylabel('|f(c)|')
plt.title('Сравнение методов по времени')
plt.grid(True, alpha=0.3, which='both')
plt.legend()
plt.show()
