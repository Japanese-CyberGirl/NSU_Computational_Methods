import math
import matplotlib.pyplot as plt
import numpy as np

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

values = [10, 100, 1000, 10000, 100000]



# Для визуализации создадим больше точек

n_values = list(range(1, 1001))  # от 1 до 1000

partial_sums = [] #значения

errors = [] #погрешность дял каждого значения



# Вычисляем частичные суммы для графика
for n in n_values:
    series_sum = alternating_harmonic_series_sum(n)
    partial_sums.append(series_sum)
    errors.append(abs(series_sum - answer))



# Выводим результаты для выбранных значений в консоль
for n in values:
    series_sum = alternating_harmonic_series_sum(n)
    error = abs(series_sum - answer)
    print(f"n = {n}: S_n = {series_sum}, погрешность = {error}")


# Дополнительная информация
print(f"\nДополнительная информация:")
print(f"Сумма первых 1000 членов: {partial_sums[-1]}")
print(f"Отличие от ln(2): {abs(partial_sums[-1] - answer)}")
print(f"Относительная погрешность: {abs(partial_sums[-1] - answer)/answer*100}%")




fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Сходимость частичных сумм
ax1.plot(n_values, partial_sums, 'b-', linewidth=1, alpha=0.7, label='Частичная сумма Sₙ')
ax1.axhline(y=answer, color='r', linestyle='--', linewidth=2, label=f'ln(2) = {answer}')
ax1.set_ylabel('Сумма Sₙ')
ax1.set_title('Сходимость ряда к ln(2)')
ax1.legend()
ax1.grid(True, alpha=0.3)



# Абсолютная погрешность
ax2.semilogy(n_values, errors, 'g-', linewidth=1, alpha=0.7, label='Абсолютная погрешность')
ax2.semilogy(n_values, [1/(n+1) for n in n_values], 'r--', linewidth=1, label='Теоретическая граница 1/(n+1)')
ax2.set_ylabel('Абсолютная погрешность (лог. шкала)')
ax2.set_title('Абсолютная погрешность приближения')
ax2.legend()
ax2.grid(True, alpha=0.3)



# Относительная погрешность
relative_errors = [error/answer for error in errors]
ax3.plot(n_values, relative_errors, 'm-', linewidth=1, alpha=0.7, label='Относительная погрешность')
ax3.semilogy(n_values, [1/(n+1) for n in n_values], 'r--', linewidth=1, label='Теоретическая граница 1/(n+1)')
ax3.set_ylabel('Относительная погрешность')
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
plt.show()


# Сходимость в логарифмическом масштабе
fig3 = plt.figure(figsize=(12, 8))
plt.semilogx(n_values, partial_sums, 'b-', linewidth=2, label='Частичная сумма Sₙ')
plt.axhline(y=answer, color='r', linestyle='--', linewidth=2, label=f'ln(2) = {answer}')
plt.xlabel('Количество членов ряда (n) - лог. шкала')
plt.ylabel('Сумма Sₙ')
plt.title('Сходимость ряда в логарифмическом масштабе')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

