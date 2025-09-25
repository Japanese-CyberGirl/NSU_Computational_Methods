import math
import matplotlib.pyplot as plt
import numpy as np

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

def regula_falsi_method(a , b , accuracy = 1e-16 , max_iterations = 10000):
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
    return root, iterations, history


a = 0.00001
b = 500000.0

root_dichotomy , iter_dichotomy , history_dichotomy = dichotomy_method(a, b)

print(f"dichotomy root = {root_dichotomy}")
print(f"dichotomy iterations = {iter_dichotomy}")

root_regula , iter_regula , history_regula = regula_falsi_method(a, b)

print(f"regula falsi root = {root_regula}")
print(f"regula falsi iterations = {iter_regula}")



fig1 = plt.figure(figsize = (12, 8))
plt.plot(history_dichotomy, [iters for iters in range(iter_dichotomy)])
plt.show()


fig2 = plt.figure(figsize = (12, 8))
plt.plot(history_regula[:-1], [iters for iters in range(iter_regula)])
plt.show()