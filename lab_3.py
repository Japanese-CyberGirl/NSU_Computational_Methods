import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def function_parabola(amount_of_segments):
    lst = []
    constanta = int(amount_of_segments//2) ** 2
    for i in range(amount_of_segments):
        value = int(i - amount_of_segments//2)
        lst.append((-1) * value**2 + constanta)
    lst[0] , lst[-1] = 0 , 0
    return lst

def function_sinus(amount_of_segments):
    lst = []
    for i in range(amount_of_segments):
        x = i / (amount_of_segments - 1) * math.pi   
        lst.append(math.sin(x))
    lst[0], lst[-1] = 0, 0  
    return lst


def function_const(amount_of_segments):
    lst = []
    for i in range(amount_of_segments):
        lst.append(1)
    lst[0] , lst[-1] = 0 , 0
    return lst


print("Введите число сегментов стержня N: ", end = "")
global N
N = int(input())
#N = 5

print()
print()

print("Введите количество 'сегментов' измерения T: ", end = "")
global t
t = int(input())
#t = 7

print()
print()

print("Введите коэффициент теплопроводности α: ", end = "")
global alpha
alpha = int(input())
#alpha = 1.0

print()
print()

print("Введите длину стержня h: ", end = "")
global h
h = int(input())
#h = 1.0

print()
print()

print("Введите размер временного сегмента в условных единицах измерения Δt: ", end = "")
global delta
delta = int(input())
#delta = 1.0

print()
print()

#start_values = function_sinus(N)
#start_values = function_parabola(N)
#start_values = function_test(N)
start_values = function_const(N)

print(start_values)

matrix = np.zeros((N , t), dtype = float)

for i in range(N):
    matrix[i][0] = start_values[i]

test_matrix = matrix.copy()

print()
np.set_printoptions(precision=3, suppress=True)
print(matrix)


print()


Lambda = (h**2)/(h**2 + 2 * alpha * delta)

theta = (alpha * delta)/(h**2 + 2 * alpha * delta)


A = np.zeros((N-2 , N - 2), dtype = float)

np.fill_diagonal(A, 1)

A[np.arange(N-3), np.arange(1, N-2)] = -theta

A[np.arange(1, N-2), np.arange(N-3)] = -theta

print(A)



def B_determination(matrix, t):
    B = []

    B.append(matrix[1][t-1] * Lambda + matrix[0][t] * theta)

    for j in range(2, N-2):
        B.append(matrix[j][t-1] * Lambda)

    B.append(matrix[-2][t-1] * Lambda + matrix[-1][t] * theta)

    return B

a = np.full(N-2, -theta)
a[0] = 0

b = np.ones(N-2)

c = np.full(N-2, -theta)
c[-1] = 0


def SLAE_run_through_method(a, b, c, d):
    n = len(d)

    alpha = np.zeros(n)
    beta  = np.zeros(n)

    gamma = b[0]
    alpha[0] = -c[0] / gamma
    beta[0]  = d[0] / gamma

    for i in range(1, n):
        gamma = b[i] + a[i] * alpha[i-1]
        if i < n-1:
            alpha[i] = -c[i] / gamma
        beta[i] = (d[i] - a[i] * beta[i-1]) / gamma

    x = np.zeros(n)
    x[-1] = beta[-1]

    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]

    return x
    
for i in range(1, t):
    vector = (SLAE_run_through_method(a, b, c, B_determination(matrix,i)))
    matrix[1:-1, i] = vector

    vector = (np.linalg.solve(A, B_determination(test_matrix,i)))
    test_matrix[1:-1, i] = vector



print()
np.set_printoptions(precision=3, suppress=True)
print(matrix)

print()
print(test_matrix)

plt.figure(figsize=(8, 5))
plt.imshow(matrix, cmap='hot', interpolation='nearest', aspect='auto')
plt.colorbar(label="Температура")
plt.xlabel("Время (t)")
plt.ylabel("Сегменты стержня")
plt.title("Heatmap распределения температуры")
plt.show()


plt.figure(figsize=(8, 5))
plt.imshow(matrix, cmap='coolwarm', interpolation='nearest', aspect='auto')
plt.colorbar(label="Температура")
plt.xlabel("Время (t)")
plt.ylabel("Сегменты стержня")
plt.title("Heatmap распределения температуры")
plt.show()
