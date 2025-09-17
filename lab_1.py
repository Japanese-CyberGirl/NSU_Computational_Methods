import numpy as np
import math

def single_epsilon():
    eps = np.float32(1.0)
    while np.float32(1.0) + eps != np.float32(1.0):
        eps_last = eps
        eps /= np.float32(2.0)
    return eps_last


def double_epsilon():
    eps = np.float64(1.0)
    while np.float64(1.0) + eps != np.float64(1.0):
        eps_last = eps
        eps /=np.float64(2.0)
    return eps_last

def exponent_range(dtype):


epsilon_32 = single_epsilon()
epsilon_64 = double_epsilon()

t_32 = -math.log2(epsilon_32)
t_64 = -math.log2(epsilon_64)

print(t_32)
print(t_64)

