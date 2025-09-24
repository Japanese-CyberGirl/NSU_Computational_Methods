import math
import matplotlib.pyplot as plt
import numpy as np
import time

def alternating_harmonic_series_sum(n):
    summa = 0.0
    for k in range(1, n + 1):
        temp = 1 / k
        summa = summa - temp if k % 2 == 0 else summa + temp
    return summa

# Точное значение
answer = math.log(2)

print("Вычисление суммы знакочередующегося гармонического ряда")
print("=" * 55)
print(f"Точное значение при n->inf: ln(2) = {answer}\n")


n_values = list(range(1, 1001))  # от 1 до 1000

partial_sums = [] #значения

errors = [] #погрешность для каждого значения

time_list = []



# Вычисляем частичные суммы для графика
for n in n_values:
    start_time = time.time()
    series_sum = alternating_harmonic_series_sum(n)
    partial_sums.append(series_sum)
    errors.append(abs(series_sum - answer))
    end_time = time.time()
    time_list.append(end_time - start_time)


# Дополнительная информация
print(f"\nДополнительная информация:")
print(f"Сумма первых 1000 членов: {partial_sums[-1]}")
print(f"Отличие от ln(2): {abs(partial_sums[-1] - answer)}")
print(f"Относительная погрешность: {abs(partial_sums[-1] - answer)/answer*100}%")
print(f"Относительная погрешность: {abs(partial_sums[-1] - answer)/answer}")


fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Сходимость частичных сумм
ax1.plot(n_values, partial_sums, 'b-', linewidth=1, alpha=0.7, label='Частичная сумма S')
ax1.axhline(y=answer, color='r', linestyle='--', linewidth=2, label=f'ln(2) = {answer}')
ax1.set_ylabel('Сумма Sₙ')
ax1.set_title('Сходимость ряда к ln(2)')
ax1.legend()
ax1.grid(True, alpha=0.3)



# Абсолютная погрешность
ax2.semilogy(n_values, errors, 'g-', linewidth=1, alpha=0.7, label='Абсолютная погрешность')
ax2.semilogy(n_values, [1/(n) for n in n_values], 'r--', linewidth=1, label='Теоретическая граница 1/(n)')
ax2.set_ylabel('Абсолютная погрешность (лог. шкала)')
ax2.set_title('Абсолютная погрешность приближения')
ax2.legend()
ax2.grid(True, alpha=0.3)



# Относительная погрешность
relative_errors = [error/answer for error in errors]
ax3.semilogy(n_values, relative_errors, 'm-', linewidth=1, alpha=0.7, label='Относительная погрешность')
ax3.semilogy(n_values, [1/(n) for n in n_values], 'r--', linewidth=1, label='Теоретическая граница 1/(n)')
ax3.set_ylabel('Относительная погрешность (лог. шкала)')
ax3.set_title('Относительная погрешность приближения')
ax3.legend()
ax3.grid(True, alpha=0.3)



# Детальный вид первых 50 членов
ax4.plot(n_values[:50], partial_sums[:50], 'b-', linewidth=2, label='Sₙ (первые 50 членов)')
ax4.axhline(y=answer, color='r', linestyle='--', linewidth=2, label=f'ln(2) = {answer}')
ax4.set_ylabel('Сумма Sₙ')
ax4.set_title('Первые 50 членов')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
#plt.show()


# Сходимость в логарифмическом масштабе
fig3 = plt.figure(figsize=(12, 8))
plt.semilogx(n_values, partial_sums, 'b-', linewidth=2, label='Частичная сумма S')
plt.axhline(y=answer, color='r', linestyle='--', linewidth=2, label=f'ln(2) = {answer}')
plt.xlabel('Количество членов ряда (n) - лог. шкала')
plt.ylabel('Сумма Sₙ')
plt.title('Сходимость ряда в логарифмическом масштабе')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
#plt.show()

fig4, ax5 = plt.subplots(figsize=(12, 8))
ax5.loglog(n_values, errors, 'g-', linewidth=1, alpha=0.7, label='Абсолютная погрешность')
ax5.loglog(n_values, [1/(n) for n in n_values], 'r--', linewidth=1, label='Теоретическая граница 1/(n)')
ax5.set_xlabel('Количество членов ряда (n) - лог. шкала')
ax5.set_ylabel('Абсолютная погрешность')
ax5.set_title('Абсолютная погрешность (лог. шкала по X)')
ax5.legend()
ax5.grid(True, alpha=0.3)
#plt.show()

fig5, ax6 = plt.subplots(figsize=(12, 8))
ax6.loglog(n_values, relative_errors, 'm-', linewidth=1, alpha=0.7, label='Относительная погрешность')
ax6.loglog(n_values, [1/(n) for n in n_values], 'r--', linewidth=1, label='Теоретическая граница 1/(n)')
ax6.set_xlabel('Количество членов ряда (n) - лог. шкала')
ax6.set_ylabel('Относительная погрешность')
ax6.set_title('Относительная погрешность (лог. шкала по X)')
ax6.legend()
ax6.grid(True, alpha=0.3)
#plt.show()

def double_epsilon():
    eps = np.float64(1.0)
    while np.float64(1.0) + eps != np.float64(1.0):
        eps_last = eps
        eps /=np.float64(2.0)
    return eps_last

epsilon = double_epsilon()

print(epsilon)


# n_values = list(range(1, 100001))  

# partial_sums = [] #значения

# errors = [] #погрешность для каждого значения

# # Вычисляем частичные суммы для графика
# for n in n_values:
#     series_sum = alternating_harmonic_series_sum(n)
#     partial_sums.append(series_sum)
#     errors.append(abs(series_sum - answer))

# cnt = 0
# for i in range(len(errors)):
#     if ((errors[i])/answer) < 10**(cnt):
#         print(f"Относительная погрешность меньше 10^{cnt} ({errors[i]/answer}) при i = {i}")
#         cnt -= 1 

# def epsilon_recursion(a, i, epsilon):
#     if (10**a) <= epsilon:
#         return i
#     return epsilon_recursion(a - 1, i * 10, epsilon)

# print(epsilon_recursion(0, 1, epsilon))


i = 1000

while (i < 100000-1):

    n_values = list(range(1, i + 1))  

    partial_sums = [] #значения

    errors = [] #погрешность для каждого значения

    start_time = time.time()

    # Вычисляем частичные суммы для графика
    for n in n_values:
        series_sum = alternating_harmonic_series_sum(n)
        partial_sums.append(series_sum)
        errors.append(abs(series_sum - answer))

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"execution time for {i} iterations = {execution_time}")

    i *= 10

# print(f"100000 iterations div 10000 iterations : {506.4003586769104/4.806163311004639}")
# print(f"10000 iterations div 1000 iterations : {4.806163311004639/0.05015087127685547}")

n_values = list(range(1, 10001))  

partial_sums = [] #значения

errors = [] #погрешность для каждого значения

time_list = []

start_time = time.time()

relative_errors = []

# Вычисляем частичные суммы для графика
for n in n_values:
    series_sum = alternating_harmonic_series_sum(n)
    partial_sums.append(series_sum)
    errors.append(abs(series_sum - answer))
    relative_errors.append(errors[-1]/answer)

    end_time = time.time()

    execution_time = end_time - start_time

    time_list.append(execution_time)

    if (len(time_list) > 1):
        time_list[-1] += time_list[-2]

# график приближения относительной погрешности к машинному эпсилон
fig10 = plt.figure(figsize=(12, 8))
plt.semilogy(relative_errors, n_values, 'b-', linewidth=2, label='Относительная погрешность к эпсилон')
plt.axhline(y=epsilon, color='r', linestyle='--', linewidth=2, label=f'epsilon = {epsilon}')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

fig11 = plt.figure(figsize=(12, 8))
plt.semilogy([i - epsilon for i in relative_errors], time_list, 'b-', linewidth=2, label='модуль разности отн. погр и эпсилон по времени')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()
