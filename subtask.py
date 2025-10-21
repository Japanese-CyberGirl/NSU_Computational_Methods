import numpy as np

def machine_epsilon_at_one(dtype):
    one = dtype(1.0)
    two = dtype(2.0)
    eps = one
    while one + eps != one:
        eps_last = eps
        eps = eps / two
    return eps_last  # eps(1) для этого dtype

def local_epsilon(x, dtype):
    # минимальный d > 0 такой, что x + d > x (чисто через деление пополам)
    one = dtype(1.0)
    two = dtype(2.0)
    d = one
    while x + d != x:
        d = d / two
    return d * two  # последнее срабатывавшее d

def emin_via_division(dtype):
    one = dtype(1.0)
    two = dtype(2.0)

    eps1 = machine_epsilon_at_one(dtype)

    x = one
    e = 0  # текущая степень относительно 1.0: x = 2^e
    emin_norm = None  # минимальная степень ДЛЯ НОРМАЛИЗОВАННЫХ
    emin_all = None   # минимальная степень (вкл. субнормалы)

    while True:
        # локальная "точность" вокруг x
        le = local_epsilon(x, dtype)

        # пока x нормализовано, должно быть le == eps1 * x
        if (emin_norm is None) and (le != eps1 * x):
            # текущий x уже субнормальный; предыдущее было последним нормализованным
            emin_norm = e  # <-- исправлено с e + 1 на e

        x_next = x / two
        if x_next == dtype(0.0):
            # текущее x — наименьшее ненулевое; его степень — минимальная "включая субнормалы"
            emin_all = e
            break

        x = x_next
        e -= 1

    return emin_all, emin_norm


# Пример использования:
emin_all_32, emin_norm_32 = emin_via_division(np.float32)
emin_all_64, emin_norm_64 = emin_via_division(np.float64)

print(f"float32: emin (включая субнормальные) = {emin_all_32}, граница нормализованных = {emin_norm_32}")
print(f"float64: emin (включая субнормальные) = {emin_all_64}, граница нормализованных = {emin_norm_64}")
