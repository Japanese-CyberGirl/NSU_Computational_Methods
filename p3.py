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
print(f"Точное значение при n->inf: ln(2) = {answer:.8f}\n")

values = [10, 100, 1000, 10000, 100000]

# Для визуализации создадим больше точек
n_values = list(range(1, 1001))  # от 1 до 1000
partial_sums = []
errors = []

# Вычисляем частичные суммы для графика
for n in n_values:
    series_sum = alternating_harmonic_series_sum(n)
    partial_sums.append(series_sum)
    errors.append(abs(series_sum - answer))

# Выводим результаты для выбранных значений
for n in values:
    series_sum = alternating_harmonic_series_sum(n)
    error = abs(series_sum - answer)
    print(f"n = {n:6d}: S_n = {series_sum:.8f}, погрешность = {error:.8f}")

# Создаем графики
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Сходимость частичных сумм
ax1.plot(n_values, partial_sums, 'b-', linewidth=1, alpha=0.7, label='Частичная сумма Sₙ')
ax1.axhline(y=answer, color='r', linestyle='--', linewidth=2, label=f'ln(2) = {answer:.6f}')
ax1.set_xlabel('Количество членов ряда (n)')
ax1.set_ylabel('Сумма Sₙ')
ax1.set_title('Сходимость знакочередующегося гармонического ряда')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Абсолютная погрешность
ax2.semilogy(n_values, errors, 'g-', linewidth=1, alpha=0.7, label='Абсолютная погрешность')
ax2.semilogy(n_values, [1/(n+1) for n in n_values], 'r--', linewidth=1, label='Теоретическая граница 1/(n+1)')
ax2.set_xlabel('Количество членов ряда (n)')
ax2.set_ylabel('Абсолютная погрешность (лог. шкала)')
ax2.set_title('Абсолютная погрешность приближения')
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Относительная погрешность в процентах
relative_errors = [error/answer * 100 for error in errors]
ax3.plot(n_values, relative_errors, 'm-', linewidth=1, alpha=0.7, label='Относительная погрешность')
ax3.set_xlabel('Количество членов ряда (n)')
ax3.set_ylabel('Относительная погрешность (%)')
ax3.set_title('Относительная погрешность приближения')
ax3.legend()
ax3.grid(True, alpha=0.3)

# График 4: Детальный вид первых 100 членов
ax5 = plt.subplot(2, 2, 4)  # Создаем четвертый subplot
ax5.plot(n_values[:100], partial_sums[:100], 'b-', linewidth=2, label='Sₙ (первые 100 членов)')
ax5.axhline(y=answer, color='r', linestyle='--', linewidth=2, label=f'ln(2) = {answer:.6f}')
ax5.set_xlabel('Количество членов ряда (n)')
ax5.set_ylabel('Сумма Sₙ')
ax5.set_title('Детальный вид: первые 100 членов')
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Дополнительная информация
print(f"\nДополнительная информация:")
print(f"Сумма первых 1000 членов: {partial_sums[-1]:.8f}")
print(f"Отличие от ln(2): {abs(partial_sums[-1] - answer):.8f}")
print(f"Относительная погрешность: {abs(partial_sums[-1] - answer)/answer*100:.4f}%")

# График колебаний вокруг точного значения
plt.figure(figsize=(12, 6))
oscillations = [s - answer for s in partial_sums]
plt.plot(n_values, oscillations, 'c-', linewidth=1, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', linewidth=1)
plt.fill_between(n_values, oscillations, 0, where=np.array(oscillations) >= 0, 
                 color='green', alpha=0.3, label='Выше ln(2)')
plt.fill_between(n_values, oscillations, 0, where=np.array(oscillations) < 0, 
                 color='red', alpha=0.3, label='Ниже ln(2)')
plt.xlabel('Количество членов ряда (n)')
plt.ylabel('Отклонение от ln(2)')
plt.title('Колебания частичных сумм вокруг точного значения')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()