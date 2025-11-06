import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def function(c):
    g = 9.8
    m = 68.1
    t = 10
    v_goal = 40
    if c == 0:
        c = 1e-20
    return (g * m / c) * (1 - math.exp((-c/m)*t)) - v_goal

def dichotomy_method(a , b , accuracy = 1e-6, max_iterations = 10000):
    if function(a) * function(b) >= 0:
        raise ValueError("Функция должна иметь разные знаки на концах интервала [a, b].")
    
    iterations = 0
    history = []

    while abs(a - b) > accuracy and iterations < max_iterations:
        midpoint = (a + b) / 2.0
        history.append(midpoint)

        if abs(function(midpoint)) < 0:
            break

        if function(a) * function(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

        iterations += 1
    
    root = (a + b) / 2.0
    return root, iterations, history

def regula_falsi_method(a , b , accuracy = 1e-6 , max_iterations = 10000):
    if function(a) * function(b) >= 0:
        raise ValueError("Функция должна иметь разные знаки на концах интервала [a, b].")
        
    iterations = 0
    history = []
    
    while iterations < max_iterations:
        c = a - function(a) * (b - a) / (function(b) - function(a))
        history.append(c)

        fc = function(c)

        if (abs(fc) < accuracy) :
            break

        if function(a) * fc < 0:
            b = c
        else:
            a = c

        iterations += 1

        if abs(b - a) < accuracy:
            break
    root = c
    return root, iterations+1, history

def regula_falsi_modified_method(x1, xu, accuracy=1e-6, max_iterations=10000):
    fl = function(x1)
    fu = function(xu)
    xr = x1
    ea = 0
    il = 0
    iu = 0
    iterations = 0
    history = []
    while True:
        xrold = xr
        xr = xu - fu * (x1 - xu) / (fl - fu)
        fr = function(xr)
        iterations += 1
        history.append(xr)
        if xr != 0:
            ea = abs((xr - xrold) / xr) * 100
        test = fl * fr
        if test < 0:
            xu = xr
            fu = function(xu)
            iu = 0
            il = il + 1
            if il >= 2:
                fl = fl / 2
        elif test > 0:
            x1 = xr
            fl = function(x1)
            il = 0
            iu = iu + 1
            if iu >= 2:
                fu = fu / 2
        else:
            ea = 0
        if ea < accuracy or iterations >= max_iterations:
            break
    return xr, iterations, history

a = 0.01
b = 50.0

root_dichotomy , iter_dichotomy , history_dichotomy = dichotomy_method(a, b)

print(f"dichotomy root = {root_dichotomy}")
print(f"dichotomy iterations = {iter_dichotomy}")
print(len(history_dichotomy))

root_regula , iter_regula , history_regula = regula_falsi_method(a, b)

print(f"regula falsi root = {root_regula}")
print(f"regula falsi iterations = {iter_regula}")
print(len(history_regula))

root_regula_mod , iter_regula_mod , history_regula_mod = regula_falsi_modified_method(a, b)

barier_d , barier_r , barier_r_m = 0 , 0 , 0

for i in range(iter_dichotomy):
    if abs(history_dichotomy[i] - history_dichotomy[-1]) < 0.1:
        barier_d = i
        break

for i in range(iter_regula):
    if abs(history_regula[i] - history_regula[-1]) < 0.1:
        barier_r = i
        break

for i in range(iter_regula_mod):
    if abs(history_regula_mod[i] - history_regula_mod[-1]) < 0.1:
        barier_r_m = i
        break


print(barier_d, barier_d, barier_r_m)

fig1 = plt.figure(figsize = (12, 8))
plt.plot([iters for iters in range(iter_dichotomy)], [abs(root_dichotomy - hist) for hist in history_dichotomy],  linewidth=2, label='приближение к ответу')
plt.axhline(0)
plt.axvline(barier_d)
plt.grid(0.1)
plt.legend()
plt.show()


fig2 = plt.figure(figsize = (12, 8))
plt.plot([iters for iters in range(iter_regula)], [abs(root_regula - hist) for hist in history_regula], linewidth=2, label='приближение к ответу')
plt.axhline(0)
plt.axvline(barier_r)
plt.grid(0.1)
plt.legend()
plt.show()

fig3 = plt.figure(figsize = (12, 8))
plt.plot([iters for iters in range(iter_regula_mod)], [abs(root_regula_mod - hist) for hist in history_regula_mod], linewidth=2, label='приближение к ответу')
plt.axhline(0)
plt.axvline(barier_r_m)
plt.grid(0.1)
plt.legend()
plt.show()



def get_barier(history, root, threshold):
    for i, val in enumerate(history):
        if abs(val - root) < threshold:
            return i + 1
    return None

# итерации до достижения точности accuracy
acc_d = len(history_dichotomy)
acc_r = len(history_regula)
acc_r_m = len(history_regula_mod)

# итерации до достижения точности 0.1
barier_d = get_barier(history_dichotomy, root_dichotomy, 0.1)
barier_r = get_barier(history_regula, root_regula, 0.1)
barier_r_m = get_barier(history_regula_mod, root_regula_mod, 0.1)

# добавим еще число итераций всего и найденные корни
data = {
"Метод": ["Дихотомия", "Regula falsi", "Regula falsi modified"],
    "Итерации до accuracy": [acc_d, acc_r, acc_r_m],
    "Итерации до 0.1": [barier_d, barier_r, barier_r_m],
    "Всего итераций": [iter_dichotomy, iter_regula, iter_regula_mod],
    "Найденный корень": [root_dichotomy, root_regula, root_regula_mod],
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
