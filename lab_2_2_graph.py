import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def original_function_1(x):
    return 4*(1-x**2) - np.exp(x)

def derivative_original_function_1(x):
    return -8*x - np.exp(x)

def original_function_2(x):
    return x**2*np.exp(x)


x_values = [int(i) for i in range(-50, 50)]
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_1 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()


x_values = [int(i) for i in range(0, 20)]
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_3 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = [int(i) for i in range(0, 10)]
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_4 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = [int(i) for i in range(0, 5)]
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_5 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = [int(i) for i in range(15, 20)]
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_6 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = np.linspace(30,33,100)
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_7 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = np.linspace(0,5,100)
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_8 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = np.linspace(-5,0,100)
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]


fig_9 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = np.linspace(-1,4,100)
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_10 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = np.linspace(-1,1,100)
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_11 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

global epsilon
epsilon = 1e-6
global max_iterations
max_iterations = 10000

def newton_method_1(x0):
    iterations = 0
    values = []
    x = x0
    f_x = original_function_1(x)
    while (iterations < max_iterations):
        iterations += 1
        f_x = original_function_1(x)
        df_x = derivative_original_function_1(x)
        if df_x == 0:
            return x, iterations, values
        x_new = x - f_x / df_x
        values.append(x_new)
        if (abs(x_new - x) < epsilon) or (abs(original_function_1(x_new)) < epsilon):
            return x_new, iterations, values
        x = x_new
    return x, iterations, values

start_time = time.time()
newton_root, newton_iterations, newton_values = newton_method_1(0)
end_time = time.time()
execution_time = end_time - start_time
newton_0_ex = execution_time
newton_iterations_array = [int(i) for i in range(newton_iterations)]

newton_function_values = [abs(original_function_1(x)) for x in newton_values]
newton_function_root = abs(original_function_1(newton_root))
print(f"Значение функции в корне, найденном методом Ньютона {newton_function_root}")

barier_n = 0
for i in range(newton_iterations):
    if newton_function_values[i] < 1e-3:
        barier_n = i
        break

fig_12 = plt.figure(figsize=(12, 8))
plt.semilogy(newton_iterations_array, newton_function_values, 'c-', linewidth=2, label='Невязка |f(xₙ)|')
plt.axvline(barier_n, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_n}')
plt.axhline(newton_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {newton_function_root:.2e}")
plt.xlabel('Итерации')
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость метода Ньютона по невязке, x₀ = 0')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приближение методом Ньютона (x₀=0) достигло невязки < 1e-3 на итерации: {barier_n}")
print(f"Время выполнения алгоритма: {execution_time}")
print()


newton_root, newton_iterations, newton_values = newton_method_1(-5)
newton_iterations_array = list(range(newton_iterations))
print(f"Значение функции в корне, найденном методом Ньютона {newton_function_root}")

newton_function_values = [abs(original_function_1(x)) for x in newton_values]
newton_function_root = abs(original_function_1(newton_root))

barier_n = 0
for i in range(newton_iterations):
    if newton_function_values[i] < 1e-3:
        barier_n = i
        break

fig_13 = plt.figure(figsize=(12, 8))
plt.semilogy(newton_iterations_array, newton_function_values, 'c-', linewidth=2, label='Невязка |f(xₙ)|')
plt.axvline(barier_n, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_n}')
plt.axhline(newton_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {newton_function_root:.2e}")
plt.xlabel('Итерации')
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость метода Ньютона по невязке, x₀ = -5')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе Ньютона для x₀ = -5 достигается на итерации {barier_n}")
print()


start_time = time.time()
newton_root, newton_iterations, newton_values = newton_method_1(20)
end_time = time.time()
execution_time = end_time - start_time
newton_20_ex = execution_time

newton_iterations_array = list(range(newton_iterations))


newton_function_values = [abs(original_function_1(x)) for x in newton_values]
newton_function_root = abs(original_function_1(newton_root))
print(f"Значение функции в корне, найденном методом Ньютона {newton_function_root}")


barier_n = 0
for i in range(newton_iterations):
    if newton_function_values[i] < 1e-3:
        barier_n = i
        break


fig_14 = plt.figure(figsize=(12, 8))
plt.semilogy(newton_iterations_array, newton_function_values, 'c-', linewidth=2,label='Невязка |f(xₙ)|')
plt.axvline(barier_n, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_n}')
plt.axhline(newton_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {newton_function_root:.2e}")
plt.xlabel('Итерации')
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость метода Ньютона по невязке, x₀ = 20')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе Ньютона для x₀ = 20 достигается на итерации {barier_n}")
print(f"Время выполнения алгоритма: {execution_time}")
print()

start_time = time.time()
newton_root, newton_iterations, newton_values = newton_method_1(10)
end_time = time.time()
execution_time = end_time - start_time
newton_iterations_array = list(range(newton_iterations))


newton_function_values = [abs(original_function_1(x)) for x in newton_values]
newton_function_root = abs(original_function_1(newton_root))
print(f"Значение функции в корне, найденном методом Ньютона {newton_function_root}")


barier_n = 0
for i in range(newton_iterations):
    if newton_function_values[i] < 1e-3:
        barier_n = i
        break


fig_15 = plt.figure(figsize=(12, 8))
plt.semilogy(newton_iterations_array, newton_function_values, 'c-', linewidth=2, label='Невязка |f(xₙ)|')
plt.axvline(barier_n, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_n}')
plt.axhline(newton_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {newton_function_root:.2e}")
plt.xlabel('Итерации')
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость метода Ньютона по невязке, x₀ = 10')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе Ньютона для x₀ = 10 достигается на итерации {barier_n}")
print(f"Время выполнения алгоритма: {execution_time}")
print()


fig_16 = plt.figure(figsize=(12, 8))
x_line = np.linspace(-2, 12, 400)
plt.plot(x_line, original_function_1(x_line), 'b-', linewidth=2, label='f(x)')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')

colors = plt.cm.viridis(np.linspace(0, 1, len(newton_values)))

for i in range(len(newton_values) - 1):
    x_i = newton_values[i]
    x_next = newton_values[i + 1]
    f_i = original_function_1(x_i)
    df_i = derivative_original_function_1(x_i)
    tangent_y = f_i + df_i * (x_line - x_i)
    color = colors[i]
    plt.plot(x_line, tangent_y, '--', linewidth=0.6, color=color)
    plt.plot([x_i, x_next], [f_i, 0], 'o--', color=color, linewidth=1)

plt.scatter(newton_root, 0, color='red', s=100, zorder=5, label='Найденный корень')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Метод Ньютона, x₀ = 10')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()   

def bisection(a, b):
    iterations = 0
    values = []
    while (b - a) > 1:
        iterations += 1
        c = (a + b) / 2
        values.append(c)
        if original_function_1(a) * original_function_1(c) < 0:
            b = c
        elif original_function_1(a) * original_function_1(c) > 0:
            a = c
        else:
            return c
    x = (a + b) / 2
    return x

start_time = time.time()
newton_root, newton_iterations, newton_values = newton_method_1(bisection(0,20))
end_time = time.time()
execution_time = end_time - start_time
newton_bis_ex = execution_time
newton_iterations_array = list(range(newton_iterations))

newton_function_values = [abs(original_function_1(x)) for x in newton_values]
newton_function_root = abs(original_function_1(newton_root))
print(f"Значение функции в корне, найденном методом Ньютона {newton_function_root}")

barier_n = 0
for i in range(newton_iterations):
    if newton_function_values[i] < 1e-3:
        barier_n = i
        break

fig_17 = plt.figure(figsize=(12, 8))
plt.semilogy(newton_iterations_array, newton_function_values, 'c-', linewidth=2, label='Невязка |f(xₙ)|')
plt.axvline(barier_n, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_n}')
plt.axhline(newton_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {newton_function_root:.2e}")
plt.xlabel('Итерации')
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость метода Ньютона')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе Ньютона для стартовой величины {bisection(0,20)} достигается на итерации под номером = {barier_n}")
print(f"Время, затраченное на работу алгоритма при выборе точки с помощью метода дихотомии = {execution_time}")

print()


def regula_falsi(a , b):
    print(f"regula falsi itertions data : ")
    iterations = 0
    values = []
    f_a = original_function_1(a)
    f_b = original_function_1(b)
    c = a

    while (b - a > epsilon) and (iterations < max_iterations):
        iterations += 1
        c = a - f_a * (b - a) / (f_b - f_a)
        values.append(c)
        f_c = original_function_1(c)

        if abs(f_c) < epsilon:
            return c , iterations , values
        if (f_a * f_c < 0):
            b = c
            f_b = f_c
        else:
            a = c
            f_a = f_c
        if (iterations > 9995):
            print(f"iterations = {iterations}, a = {a}, b = {b}, c = {c}, f_c = {f_c}")
    if (iterations == 10000):
        print("Алгоритм завершил работу из-за ограничения по лимиту итераций")

    return c , iterations , values

start_time = time.time()
regula_root, regula_iterations, regula_values = regula_falsi(0, 20)
end_time = time.time()
execution_time = end_time - start_time
reg_ex = execution_time
regula_iterations_array = list(range(regula_iterations))

regula_function_values = [abs(original_function_1(x)) for x in regula_values]
regula_function_root = abs(original_function_1(regula_root))
print(f"Значение функции в корне, найденном методом regula falsi {regula_function_root}")

barier_r = 0
for i in range(regula_iterations):
    if regula_function_values[i] < 1e-3:
        barier_r = i
        break

fig_18 = plt.figure(figsize=(12, 8))
plt.semilogy(regula_iterations_array, regula_function_values, 'g-', linewidth=2, label='Невязка |f(xₙ)| (regula falsi)')
plt.axvline(barier_r, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_r}')
plt.axhline(regula_function_root, color='r', linestyle='--', linewidth=2,label=f"Невязка в корне = {regula_function_root:.2e}")
plt.xlabel('Итерации')
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость метода regula falsi по невязке')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе regula falsi достигается на итерации под номером = {barier_r}")
print(f"Время, затраченное на работу алгоритмы при визуальном выборе отрезке = {execution_time}")
print()



def mod_regula_falsi(a, b):
    iterations = 0
    values = []
    f_a = original_function_1(a)
    f_b = original_function_1(b)
    c = a
    i_a = 0
    i_b = 0

    while (b - a) > epsilon:
        iterations += 1
        c = b - f_b * (b - a) / (f_b - f_a)
        values.append(c)
        f_c = original_function_1(c)

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

start_time = time.time()
mod_root, mod_iterations, mod_values = mod_regula_falsi(0, 20)
end_time = time.time()
execution_time = end_time - start_time
mod_ex = execution_time

mod_iterations_array = [int(i) for i in range(mod_iterations)]


mod_function_values = [abs(original_function_1(x)) for x in mod_values]
mod_function_root = abs(original_function_1(mod_root))
print(f"Значение функции в корне, найденном методом regula falsi modified {mod_function_root}")


barier_m = 0
for i in range(mod_iterations):
    if mod_function_values[i] < 1e-3:
        barier_m = i
        break

fig_19 = plt.figure(figsize=(12, 8))
plt.semilogy(mod_iterations_array, mod_function_values, 'm-', linewidth=2, label='Невязка |f(xₙ)| (mod regula falsi)')
plt.axvline(barier_m, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_m}')
plt.axhline(mod_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {mod_function_root:.2e}")
plt.xlabel('Итерации')
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость методом модифицированного regula falsi по невязке')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе mod regula falsi достигается на итерации под номером = {barier_m}")
print(f"Время, затраченное на работу алгоритмы при визуальном выборе отрезке = {execution_time}")
print()


def secant_method(x0, x1):
    iterations = 0
    values = []
    x_current = x1
    x_previous = x0
    while (abs(x_current  - x_previous) > epsilon) and (iterations < max_iterations):
        iterations += 1
        f_c = original_function_1(x_current)
        f_p = original_function_1(x_previous)
        numerator = x_current - x_previous
        denominator = f_c - f_p
        product = (numerator/denominator)*f_c
        x_new = x_current - product
        values.append(x_new)
        if (abs(x_new - x_current) < epsilon) or (abs(original_function_1(x_new)) < epsilon):
            return x_new, iterations, values
        x_previous = x_current
        x_current = x_new
    return x_current, iterations, values

start_time = time.time()
secant_root, secant_iterations, secant_values = secant_method(0, 0.001)
end_time = time.time()
execution_time = end_time - start_time
sec_ex = execution_time

secant_iterations_array = [int(i) for i in range(secant_iterations)]


secant_function_values = [abs(original_function_1(x)) for x in secant_values]
secant_function_root = abs(original_function_1(secant_root))
print(f"Значение функции в корне, найденном методом секущих {secant_function_root}")


barier_s = 0
for i in range(secant_iterations):
    if secant_function_values[i] < 1e-3:
        barier_s = i
        break

fig_20 = plt.figure(figsize=(12, 8))
plt.semilogy(secant_iterations_array, secant_function_values, 'm-', linewidth=2, label='Невязка |f(xₙ)| (secant method)')

plt.xlabel('Итерации')
plt.axvline(barier_s, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_s}')
plt.axhline(secant_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {secant_function_root:.2e}")
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость метода секущих по невязке')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"Приблизительный ответ в методе секущих достигается на итерации под номером = {barier_s}")
print(f"Время, затраченное на работу алгоритмы при визуальном выборе отрезке = {execution_time}")

print()

def Steffensen_method(x0):
    iterations = 0
    values = []
    x = x0
    f_x = original_function_1(x)
    while (abs(f_x) > epsilon) and (iterations < max_iterations):
        iterations += 1
        f_x = original_function_1(x)
        numerator = f_x
        denominator = original_function_1(x + f_x) - f_x
        product = numerator / denominator * f_x
        x_new = x - product
        values.append(x_new)
        if (abs(x_new - x) < epsilon) or (abs(original_function_1(x_new)) < epsilon):
            return x_new, iterations, values
        x = x_new
    return x, iterations, values

start_time = time.time()
Stef_root, Stef_iterations, Stef_values = Steffensen_method(0)
end_time = time.time()
execution_time = end_time - start_time
St_ex = execution_time

Stef_iterations_array = [int(i) for i in range(Stef_iterations)]


Stef_function_values = [abs(original_function_1(x)) for x in Stef_values]
Stef_function_root = abs(original_function_1(Stef_root))
print(f"Значение функции в корне, найденном методом Стеффенсена {Stef_function_root}")


barier_St = 0
for i in range(Stef_iterations):
    if Stef_function_values[i] < 1e-3:
        barier_St = i
        break

fig_21 = plt.figure(figsize=(12, 8))
plt.semilogy(Stef_iterations_array, Stef_function_values, 'm-', linewidth=2, label='Невязка |f(xₙ)| (Steffensen)')
plt.xlabel('Итерации')
plt.axvline(barier_St, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_St}')
plt.axhline(Stef_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {Stef_function_root:.2e}")
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость методом Стеффенсена')
plt.legend()
plt.grid(True, alpha=0.3, which="both")
plt.show()

print(f"Приблизительный ответ в методе Стеффенсена достигается на итерации под номером = {barier_St}")
print(f"Время, затраченное на работу алгоритмы при визуальном выборе отрезке = {execution_time}")

print()

print("Сравнение методов")

compare_table = []

compare_table.append((
    "Ньютон (x₀ = 0)",
    newton_method_1(0)[1],
    newton_method_1(0)[0],
    abs(original_function_1(newton_method_1(0)[0])),
    newton_0_ex
))

compare_table.append((
    "Ньютон (x₀ = 20)",
    newton_method_1(20)[1],
    newton_method_1(20)[0],
    abs(original_function_1(newton_method_1(20)[0])),
    newton_20_ex
))

compare_table.append((
    "Ньютон x Дихотомия",
    newton_method_1(bisection(0,20))[1],
    newton_method_1(bisection(0,20))[0],
    abs(original_function_1(newton_method_1(bisection(0,20))[0])),
    newton_bis_ex
))

compare_table.append((
    "Regula Falsi",
    regula_iterations,
    regula_root,
    abs(original_function_1(regula_root)),
    reg_ex
))

compare_table.append((
    "Mod. Regula Falsi",
    mod_iterations,
    mod_root,
    abs(original_function_1(mod_root)),
    mod_ex
))

compare_table.append((
    "Секущих",
    secant_iterations,
    secant_root,
    abs(original_function_1(secant_root)),
    sec_ex
))

compare_table.append((
    "Стеффенсена",
    Stef_iterations,
    Stef_root,
    abs(original_function_1(Stef_root)),
    St_ex
))

compare_table.sort(key=lambda x: x[4])

df = pd.DataFrame(
    compare_table,
    columns=["Метод", "Итерации", "Корень", "|f(x)|", "Время (сек)"]
)

print(df.to_string(index=False))
print()


for i in np.linspace(0, 1, 100):
    print(f"x0 = {i}")
    print(f"x0 = {i} , root = {Steffensen_method(i)[0]} , iterations = {Steffensen_method(i)[1]}")

x_values = [int(i) for i in range(340, 360)]
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_22 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

x_values = [int(i) for i in range(333, 335)]
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_23 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()


x_values = [int(i) for i in range(0, 380)]
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_24 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Корень уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()


start_time = time.time()
Stef_root, Stef_iterations, Stef_values = Steffensen_method(0.93)
end_time = time.time()
execution_time = end_time - start_time
St_ex = execution_time

Stef_iterations_array = [int(i) for i in range(Stef_iterations)]


Stef_function_values = [abs(original_function_1(x)) for x in Stef_values]
Stef_function_root = abs(original_function_1(Stef_root))
print(f"Значение функции в корне, найденном методом Стеффенсена при x₀ = 0.93 : {Stef_function_root}")

barier_St = 0
for i in range(Stef_iterations):
    if Stef_function_values[i] < 1e-3:
        barier_St = i
        break

fig_25 = plt.figure(figsize=(12, 8))
plt.semilogy(Stef_iterations_array, Stef_function_values, 'm-', linewidth=2, label='Невязка |f(xₙ)| (Steffensen)')
plt.xlabel('Итерации')
plt.axvline(barier_St, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_St}')
plt.axhline(Stef_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {Stef_function_root:.2e}")
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость методом Стеффенсена, x₀ = 0.93')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()


start_time = time.time()
Stef_root, Stef_iterations, Stef_values = Steffensen_method(0.95)
end_time = time.time()
execution_time = end_time - start_time
St_ex = execution_time

Stef_iterations_array = [int(i) for i in range(Stef_iterations)]


Stef_function_values = [abs(original_function_1(x)) for x in Stef_values]
Stef_function_root = abs(original_function_1(Stef_root))
print(f"Значение функции в корне, найденном методом Стеффенсена при x₀ = 0.95 : {Stef_function_root}")


barier_St = 0
for i in range(Stef_iterations):
    if Stef_function_values[i] < 1e-3:
        barier_St = i
        break

fig_26 = plt.figure(figsize=(12, 8))
plt.semilogy(Stef_iterations_array, Stef_function_values, 'm-', linewidth=2, label='Невязка |f(xₙ)| (Steffensen)')
plt.xlabel('Итерации')
plt.axvline(barier_St, color='orange', linestyle='--', linewidth=2, label=f'|f(xₙ)| < 1e-3 на итерации {barier_St}')
plt.axhline(Stef_function_root, color='r', linestyle='--', linewidth=2, label=f"Невязка в корне = {Stef_function_root:.2e}")
plt.ylabel('|f(xₙ)|')
plt.title('Сходимость методом Стеффенсена, x₀ = 0.95')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()




def Steffensen_method_data(x0):
    iterations = 0
    values = []
    x = x0
    f_x = original_function_1(x)
    while (abs(f_x) > epsilon) and (iterations < max_iterations):
        iterations += 1
        f_x = original_function_1(x)
        numerator = f_x
        
        denominator = original_function_1(x + f_x) - f_x
        product = numerator / denominator * f_x
        x_new = x - product
        if (iterations < 15):
            print()
            print(f"iteration = {iterations}")
            print(f"numerator = {f_x}")
            print(f"denominator = {denominator}")
            print(f"product = {product}")
            print(f"x_new = {x_new}")
            print()
        values.append(x_new)
        if (abs(x_new - x) < epsilon) or (abs(original_function_1(x_new)) < epsilon):
            return x_new, iterations, values
        x = x_new
    return x, iterations, values

print()
print()
print()
print("Метод Стеффенсена для 0.89")
Steffensen_method_data(0.89)
print()
print()
print()
print("Метод Стеффенсена для 0.93")
Steffensen_method_data(0.93)
print()
print()
print()
print("Метод Стеффенсена для 0.95")
Steffensen_method_data(0.95)



newton_root, newton_iterations, newton_values = newton_method_1(0)
newton_bis_root, newton_bis_iterations, newton_bis_values = newton_method_1(bisection(0, 20))
#regula_root, regula_iterations, regula_values = regula_falsi(0, 20)
mod_root, mod_iterations, mod_values = mod_regula_falsi(0, 20)
secant_root, secant_iterations, secant_values = secant_method(0, 0.001)
Stef_root, Stef_iterations, Stef_values = Steffensen_method(0)

methods_data = {
    "newton": newton_values,
    "newton + bisection": newton_bis_values,
    #"regula Falsi": regula_values,
    "mod. regula Falsi": mod_values,
    "secant": secant_values,
    "Steffensen": Stef_values
}

# roots = {
#     "newton": newton_root,
#     "newton + bisection": newton_bis_root,
#     #"regula falsi": regula_root,
#     "mod. regula falsi": mod_root,
#     "secant": secant_root,
#     "Steffensen": Stef_root
# }

fig_27 = plt.figure(figsize=(12, 8))

errors_newton = [abs(original_function_1(x)) for x in newton_values]
errors_newton_bis = [abs(original_function_1(x)) for x in newton_bis_values]
errors_mod = [abs(original_function_1(x)) for x in mod_values]
errors_secant = [abs(original_function_1(x)) for x in secant_values]
errors_stef = [abs(original_function_1(x)) for x in Stef_values]

plt.plot(range(len(errors_newton)), errors_newton, label='newton', linewidth=2)
plt.plot(range(len(errors_newton_bis)), errors_newton_bis, label='newton + bisection', linewidth=2)
plt.plot(range(len(errors_mod)), errors_mod, label='Mod. regula falsi', linewidth=2)
plt.plot(range(len(errors_secant)), errors_secant, label='secant', linewidth=2)
plt.plot(range(len(errors_stef)), errors_stef, label='Steffensen', linewidth=2)

plt.yscale('log')
plt.xlabel('Количество итераций')
plt.ylabel('Ошибка |f(xₙ)|')
plt.title('Зависимость ошибки |f(xₙ)| от количества итераций для различных методов')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()



errors_newton = [abs(original_function_1(x)) for x in newton_values]
times_newton = np.linspace(0, newton_0_ex, len(errors_newton))


errors_newton_bis = [abs(original_function_1(x)) for x in newton_bis_values]
times_newton_bis = np.linspace(0, newton_bis_ex, len(errors_newton_bis))


errors_mod = [abs(original_function_1(x)) for x in mod_values]
times_mod = np.linspace(0, mod_ex, len(errors_mod))


errors_sec = [abs(original_function_1(x)) for x in secant_values]
times_sec = np.linspace(0, sec_ex, len(errors_sec))


errors_stef = [abs(original_function_1(x)) for x in Stef_values]
times_stef = np.linspace(0, St_ex, len(errors_stef))




plt.figure(figsize=(12, 8))
plt.plot(times_newton, errors_newton, linewidth=2, label='Newton')
plt.plot(times_newton_bis, errors_newton_bis, linewidth=2, label='Newton + bisection')
plt.plot(times_mod, errors_mod, linewidth=2, label='Mod. regula falsi')
plt.plot(times_sec, errors_sec, linewidth=2, label='Secant')
plt.plot(times_stef, errors_stef, linewidth=2, label='Steffensen')
plt.yscale("log")
plt.xlabel("Время (сек)")
plt.ylabel("|f(xₙ)|")
plt.title("Зависимость ошибки |f(xₙ)| от времени для различных методов")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()


# def Steffensen_method_data_2(x0):
#     iterations = 0
#     values = []
#     x = x0
#     f_x = original_function_1(x)
#     while (abs(f_x) > epsilon) and (iterations < max_iterations):
#         iterations += 1
#         f_x = original_function_1(x)
#         numerator = f_x
        
#         denominator = original_function_1(x + f_x) - f_x
#         product = numerator / denominator * f_x
#         x_new = x - product
#         print()
#         print(f"iteration = {iterations}")
#         print(f"numerator = {f_x}")
#         print(f"denominator = {denominator}")
#         print(f"product = {product}")
#         print(f"x_new = {x_new}")
#         print()
#         values.append(x_new)
#         if (abs(x_new - x) < epsilon) or (abs(original_function_1(x_new)) < epsilon):
#             return x_new, iterations, values
#         x = x_new
#     return x, iterations, values

# print()
# print()
# print("Strange Steffenson:")
# print()
# Steffensen_method_data_2(0.9494949494949496)
