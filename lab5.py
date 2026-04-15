import math
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return 50 + 20 * math.sin(math.pi * x / 12) + 5 * math.exp(-0.2 * (x - 12)**2)

a = 0
b = 24

# значення I0 
I0, _ = quad(f, a, b)
print(f"Точне значення I0: {I0}")

# Сімпсона
def simpson(a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    
    sum_odd = 0
    for i in range(1, int(N), 2):
        sum_odd += f(a + i * h)
        
    sum_even = 0
    for i in range(2, int(N)-1, 2):
        sum_even += f(a + i * h)
        
    return (h / 3) * (f(a) + 4 * sum_odd + 2 * sum_even + f(b))

h_v = []
e_simp = []
e_runge = []
e_aitken = []
e_teor = [] # O(h^4)

N_vals = [8, 16, 32, 64, 128, 256, 512] 

for N in N_vals:
    h = (b - a) / N
    h_v.append(h)
    
    #  Сімпсон
    I_N = simpson(a, b, N)
    e_simp.append(abs(I_N - I0))
    
    # Рунге-Ромберг
    I_N_2 = simpson(a, b, int(N / 2))
    I_R = I_N + (I_N - I_N_2) / 15
    e_runge.append(abs(I_R - I0))
    
    # Ейткен
    I_N_4 = simpson(a, b, int(N / 4))
    znam = 2 * I_N_2 - (I_N + I_N_4)
    if znam != 0:
        I_E = (I_N_2**2 - I_N * I_N_4) / znam
    else:
        I_E = I_N
    e_aitken.append(abs(I_E - I0))
    
    # 4. Теоретична лінія
    e_teor.append(h**4)


N_values = []
errors_simp_N = []
N_opt = 0

for N in range(10, 1002, 2):
    I_N = simpson(a, b, N)
    err = abs(I_N - I0)
    
    N_values.append(N)
    errors_simp_N.append(err)
    
    if err <= 1e-12 and N_opt == 0:
        N_opt = N

print(f"Оптимальне N_opt (похибка < 1e-12): {N_opt}")




# ГРАФІК 1
plt.figure()
plt.loglog(h_v, e_simp, marker='o', color='red', label='Сімпсон')
plt.loglog(h_v, e_runge, marker='s', color='green', label='Рунге-Ромберг')
plt.loglog(h_v, e_aitken, marker='^', color='purple', label='Ейткен')
plt.loglog(h_v, e_teor, marker='', color='black', linestyle='--', label='Теоретична O(h^4)')

plt.title("Порівняння похибок методів")
plt.xlabel("Крок h")
plt.ylabel("Похибка (логарифмічна шкала)")
plt.legend()
plt.grid()
plt.show()

# ГРАФІК 2
plt.figure()
plt.plot(N_values, errors_simp_N, color='blue')
plt.yscale('log') # Робимо вісь Y логарифмічною, щоб було видно падіння похибки
plt.title("Дослідження точності: Похибка Сімпсона від N")
plt.xlabel("Кількість розбиттів (N)")
plt.ylabel("Похибка |I(N) - I0|")
plt.grid()
plt.show()

# ГРАФІК 3
plt.figure()
x_vals = [i * 0.1 for i in range(241)]
y_vals = [f(x) for x in x_vals]
plt.plot(x_vals, y_vals, color='orange')
plt.title("Базовий графік функції")
plt.xlabel("x (год)")
plt.ylabel("f(x)")
plt.grid()
plt.show()

calls = 0

def adaptive_simpson(a, b, eps):
    global calls
    h = b - a
    c = (a + b) / 2
    
    fa = f(a)
    fb = f(b)
    fc = f(c)
    calls += 3
    S1 = (h / 6) * (fa + 4 * fc + fb)
    
    d = (a + c) / 2
    e = (c + b) / 2
    fd = f(d)
    fe = f(e)
    calls += 2
    S2 = (h / 12) * (fa + 4 * fd + fc) + (h / 12) * (fc + 4 * fe + fb)
    
    if abs(S1 - S2) <= 15 * eps:
        return S2 + (S2 - S1) / 15
    else:
        return adaptive_simpson(a, c, eps / 2) + adaptive_simpson(c, b, eps / 2)

I_adapt = adaptive_simpson(a, b, 1e-6)
print(f"Адаптивний алгоритм: {I_adapt}")
print(f"Кількість викликів функції: {calls}")