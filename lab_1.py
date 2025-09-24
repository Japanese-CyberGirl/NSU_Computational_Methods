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
   e_max = 0
   value = dtype(1.0)
   while True:
       test_value = value * dtype(2.0)
       if (np.isinf(test_value)):
           break
       value = test_value
       e_max += 1
   return e_max + 1
    
#    e_min = 0
#    value = dtype(1.0)
#    while True:
#        test_value = value / dtype(2.0)
#        if (test_value == dtype(0.0)):
#            break 
#        value = test_value
#        e_min -= 1
       
#    return e_min, e_max + 1 #так как нас интересует именно предпоследняя итерация

def actual_exponent_range(dtype):
    info = np.finfo(dtype)
    return info.minexp , info.maxexp

def mathematical_exponent_range(k):
    bias = 2**(k-1)-1
    emin = 1 - bias
    return emin

print("Epsilons:")
epsilon_32 = single_epsilon()
epsilon_64 = double_epsilon()

print(f"epsilon_32 = {epsilon_32}")
print(f"epsilon_64 = {epsilon_64}")

print("the bytes amount:")
t_32 = -math.log2(epsilon_32)
t_64 = -math.log2(epsilon_64)

print(t_32)
print(t_64)

print("Exponents - '1' \n")
e_max_32 = exponent_range(np.float32)
e_max_64 = exponent_range(np.float64)

actual_e_min_32 , actual_e_max_32 = (actual_exponent_range(np.float32))
actual_e_min_64 , actual_e_max_64 = (actual_exponent_range(np.float64))

print(f"e_min32 = {actual_e_min_32}")
print(f"e_min64 = {actual_e_min_64}")
print(f"e_max32 = {actual_e_max_32}")
print(f"e_max64 = {actual_e_max_64}")



# print(f"experemetnal e_min_32 = {e_min_32} || actual e_min_32 = {actual_e_min_32}")
# print(f"experemental e_max_32 = {e_max_32} || actual e_max_32 = {actual_e_max_32}")
# print(f"experemental e_min_64 = {e_min_64} || actual e_min_64 = {actual_e_min_64}")
# print(f"experemental e_max_64 = {e_max_64} || actual e_max_64 = {actual_e_max_64}")

print(f"experement e_max_32 = {e_max_32}")
print(f"experement e_max_64 = {e_max_64}")

print(f"mathematica emin_32 = {mathematical_exponent_range(8)}")
print(f"mathematica emin_64 = {mathematical_exponent_range(11)}")

print("First comparing")

print(np.float32(1.0))
print(np.float32(1.0) + np.float32(epsilon_32)/2)
print(np.float32(1.0) + np.float32(epsilon_32))
print(np.float32(1.0) + np.float32(epsilon_32) + np.float32(epsilon_32)/2) #???

print("First comparing double pr")

print(np.float64(1.0))
print(np.float64(1.0) + np.float64(epsilon_64)/2)
print(np.float64(1.0) + np.float64(epsilon_64))
print(np.float64(1.0) + np.float64(epsilon_64) + np.float64(epsilon_64)/2) #???

#print(np.float32(epsilon_32)/2)


print("Last comparing")

a , b = np.float32(1.0) , np.float32(10**(-16))

print("float32:")

print(f"a + b = {a+b}")
d = a + b
print(f"(a + b) + b = {d + b}")

print(f"b + b = {b + b}")
e = (b + b)
print(f"a + (b + b) = {a + e}")


print("float64:")

a , b = np.float64(1.0) , np.float64(10**(-16))


print(f"a + b = {a+b}")
d = a + b
print(f"(a + b) + b = {d + b}")

print(f"b + b = {b + b}")
e = (b + b)
print(f"a + (b + b) = {a + e}")

print(f"epsilon_32 = {epsilon_32}")
print(f"epsilon_64 = {epsilon_64}")

print(f"1 + epsilon_64 = {np.float64(1.0) + np.float64(epsilon_64)}")
