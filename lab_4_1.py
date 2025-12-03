import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def original_function(x):
    return 1 / (1 + np.exp(x))


def left_rechtangle(a, b, n):
    h = (b - a) / n
    s = 0.0
    for k in range(n):
        x = a + k * h
        s += original_function(x)
    return h * s


def right_rechtangle(a, b, n):
    h = (b - a) / n
    s = 0.0
    for k in range(1, n + 1):
        x = a + k * h
        s += original_function(x)
    return h * s


def middle_rechtangle(a, b, n):
    h = (b - a) / n
    s = 0.0
    for k in range(n):
        x = a + (k + 0.5) * h
        s += original_function(x)
    return h * s

def trapezoid(a, b, n):
    h = (b - a) / n
    s = 0.5 * (original_function(a) + original_function(b))
    for k in range(1, n):
        x = a + k * h
        s += original_function(x)
    return h * s

def simpson(a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even")
    
    h = (b - a) / n
    s = original_function(a) + original_function(b)

    for k in range(1, n):
        x = a + k * h
        if k % 2 == 0:
            s += 2 * original_function(x)
        else:
            s += 4 * original_function(x)
    return (h / 3) * s


        