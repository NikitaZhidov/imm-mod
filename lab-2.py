import numpy as np
from scipy.stats import f_oneway

alpha = 0.05

n_group = 100 # длина выборки
N = 1000 # Кол-во экспериментов

avg = 1
delta = 0.1

def has_difference(expectations):
    samples = [np.random.normal(e, avg, n_group) for e in expectations]
    _, p_value = f_oneway(*samples)
    return p_value < alpha

# средние для групп
expectations = [avg, avg + delta, avg - delta]

has_diff = sum(has_difference(expectations) for _ in range(N))

print("Мощность критерия:",  1 - has_diff / N)
