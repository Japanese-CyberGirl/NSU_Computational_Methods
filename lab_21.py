import math
import matplotlib.python as plt
import numpy as np

def init_function(c):
    g = 9.8
    m = 68.1
    t = 10
    v_goal = 40
    if c == 0:
        c = 1e-20
    return (g * m / c) * (1 - math.exp((-c/m)*t))


