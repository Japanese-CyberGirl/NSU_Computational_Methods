import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def original_function_1(x):
    return 4*(1-x**2) - np.exp(x)

def original_function_2(x):
    return x**2*np.exp(x)

x_values = np.linspace(-50, 50, 1000)
y_values_1 = [original_function_1(x) for x in x_values]
y_values_2 = [original_function_2(x) for x in x_values]

fig_1 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_1, 'b-', linewidth=2, label='f(x) = 4(1 - x²) - eˣ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = 4(1 - x²) - eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

fig_2 = plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values_2, 'g-', linewidth=2, label='f(x) = x² * eˣ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции f(x) = x² * eˣ')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()
