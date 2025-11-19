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
    return lst

