import math
import matplotlib as plt
import numpy as np

def alternating_harmonic_series_sum(n):

    summa = 0.0
    for k in range(1, n + 1):
        temp = 1 / k
        summa = summa - temp if k % 2 == 0 else summa + temp
    return summa



answer = math.log(2)

print("Вычисление суммы знакочередующегося гармонического ряда")
print("=" * 55)
print(f"Точное значение при n->inf: ln(2) : {answer}\n")

values = [10, 100, 1000, 10000, 100000]

for n in values:
    series_sum = alternating_harmonic_series_sum(n)
    print(f"n = {n}: S_n = {series_sum}")

