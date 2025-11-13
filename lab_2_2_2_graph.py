import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def original_function_2(x):
    return x**2*np.exp(x)

def derivative_original_function_2(x):
    return x**2*np.exp(x) + 2*x*np.exp(x)

global epsilon
epsilon = 1e-6
global max_iterations
max_iterations = 10000

x_values = np.linspace(-50, 50, 1000)
y_values_2 = [original_function_2(x) for x in x_values]

fig_1 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_2, 'g-', linewidth=2, label='f(x) = x² * eˣ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = x² * eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()


x_values = np.linspace(-5, 5, 1000)
y_values_2 = [original_function_2(x) for x in x_values]

fig_2 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_2, 'g-', linewidth=2, label='f(x) = x² * eˣ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = x² * eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()


x_values = np.linspace(0, 5, 1000)
y_values_2 = [original_function_2(x) for x in x_values]

fig_3 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_2, 'g-', linewidth=2, label='f(x) = x² * eˣ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = x² * eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()


x_values = np.linspace(-1, 2, 1000)
y_values_2 = [original_function_2(x) for x in x_values]

fig_4 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_2, 'g-', linewidth=2, label='f(x) = x² * eˣ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = x² * eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()


x_values = np.linspace(-2, 1, 1000)
y_values_2 = [original_function_2(x) for x in x_values]

fig_5 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_2, 'g-', linewidth=2, label='f(x) = x² * eˣ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = x² * eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

def newton_method_2(x0):
    iterations = 0
    values = []
    x = x0
    f_x = original_function_2(x)
    while (iterations < max_iterations):
        iterations += 1
        f_x = original_function_2(x)
        df_x = derivative_original_function_2(x)
        if df_x == 0:
            return x, iterations, values
        x_new = x - f_x / df_x
        values.append(x_new)
        if (abs(x_new - x) < epsilon) or (abs(original_function_2(x_new)) < epsilon):
            return x_new, iterations, values
        x = x_new
    return x, iterations, values


start_time = time.time()
newton_root, newton_iterations, newton_values = newton_method_2(x0=1)
end_time = time.time()
execution_time = end_time - start_time
newton_iterations_array = [int(i) for i in range(newton_iterations)]

newton_function_values = [abs(original_function_2(x)) for x in newton_values]
newton_function_root = abs(original_function_2(newton_root))

barier_n = 0
for i in range(newton_iterations):
    if newton_function_values[i] < 1e-3:
        barier_n = i
        break

fig_6 = plt.figure(figsize=(12, 8))
plt.semilogy(newton_iterations_array, newton_function_values, 'c-', linewidth=2, label='Невязка |f(xₙ)|')
plt.xlabel('Итерации')
plt.axvline(barier_n, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_n}')
plt.axhline(newton_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {newton_function_root:.2e}")
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость метода Ньютона по невязке')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе Ньютона для стартовой величины {1} достигается на итерации под номером = {barier_n}")
print(f"Время, затраченное на работу алгоритма при выборе точки x0=1 = {execution_time}")
print()

print()

def explicit_newton_method(x0, m):
    iterations = 0
    values = []
    x = x0
    f_x = original_function_2(x)
    while (iterations < max_iterations):
        iterations += 1
        f_x = original_function_2(x)
        df_x = derivative_original_function_2(x)
        if df_x == 0:
            return x, iterations, values
        x_new = x - m * f_x / df_x
        values.append(x_new)
        if (abs(x_new - x) < epsilon) or (abs(original_function_2(x_new)) < epsilon):
            return x_new, iterations, values
        x = x_new
    return x, iterations, values

start_time = time.time()
exp_newton_root, exp_newton_iterations, exp_newton_values = explicit_newton_method(x0=1, m=2)
end_time = time.time()
execution_time = end_time - start_time
exp_newton_iterations_array = [int(i) for i in range(exp_newton_iterations)]

exp_newton_function_values = [abs(original_function_2(x)) for x in exp_newton_values]
exp_newton_function_root = abs(original_function_2(exp_newton_root))

barier_exp_n = 0
for i in range(exp_newton_iterations):
    if exp_newton_function_values[i] < 1e-3:
        barier_exp_n = i
        break

fig_7 = plt.figure(figsize=(12, 8))
plt.semilogy(exp_newton_iterations_array, exp_newton_function_values, 'c-', linewidth=2, label='Невязка |f(xₙ)| для уточнённого Ньютона')
plt.xlabel('Итерации')
plt.axvline(barier_exp_n, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_exp_n}')
plt.axhline(exp_newton_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {exp_newton_function_root:.2e}")
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость уточнённого метода Ньютона по невязке')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в уточненным методе Ньютона для стартовой величины {1} и m = {2} достигается на итерации под номером = {barier_exp_n}")
print(f"Время, затраченное на работу алгоритма при выборе точки x0=1 и m=2 = {execution_time}")
print()

print()
