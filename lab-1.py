from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import numpy as np
from scipy.special import erf
from scipy.stats import norm

N = 100
V = 690

# Определяем функцию плотности распределения
def fn(x, A, K):
    return A * x**3 * (K - x)

K = 10  # Значение параметра K, максимально возможная страховая выплата
lower_limit = 0

# Находим значение константы A, чтобы интеграл от функции f был равен 1
integral_value, _ = quad(fn, lower_limit, K, args=(1, K))  # Интегрирование функции f от lower_limit до upper_limit

A = 1 / integral_value  # Находим значение константы A


# Найти Максимум
M = max([fn(x, A, K) for x in np.linspace(0, K + 1, 100000)])

# # Генерируем для клиента по методу Неймана
def generate_payments():
    global N, K, M

    payments = []

    while len(payments) < N:
        u1 = np.random.rand()
        u2 = np.random.rand()

        x1 = K * u1
        x2 = M * u2

        if (x2 <= fn(x1, A, K)):
            payments.append(x1)

    return payments


n = 1000
successful_payments = 0;
for i in range(n):
    successful_payments += 1 if sum(generate_payments()) < V else 0

print(f"Вероятность разориться: {1 - successful_payments / n} если резерв: {V}, максимальная выплата: {K}, кол-во клиентов: {N}")

# Считаем апроксимацию
expectation_math, _ = quad(lambda x: x * fn(x, A, K), lower_limit, K)
variance, _ = quad(lambda x: (x - expectation_math) ** 2 * fn(x, A, K), lower_limit, K)


def laplace_val(x):
    return norm.cdf(x) - 0.5


x = (V - N*expectation_math)/np.sqrt(N*variance)

print(f"Значение, полученное с помощью апроксимации: {1/2 - laplace_val(x)}")
