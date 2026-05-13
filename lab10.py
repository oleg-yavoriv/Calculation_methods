import numpy as np
import matplotlib.pyplot as plt
import math


def f(x, y):
    return x - y + 1


def exact_solution(x):
    return x + math.exp(-x)


def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def adams_pc2_step(x_prev, y_prev, x_curr, y_curr, h):

    y_pred = y_curr + (h/2) * (3*f(x_curr, y_curr) - f(x_prev, y_prev))

    y_corr = y_curr + (h/2) * (f(x_curr + h, y_pred) + f(x_curr, y_curr))
    return y_pred, y_corr


# дані
a, b = 0.0, 2.0
y0 = 1.0
h_const = 0.01
eps = 1e-4

# Адамса 
x_adams = np.arange(a, b + h_const, h_const)
y_adams = np.zeros(len(x_adams))
y_exact_adams = np.zeros(len(x_adams))
error_adams_local = np.zeros(len(x_adams))
error_adams_est = np.zeros(len(x_adams))

y_adams[0] = y0
y_exact_adams[0] = exact_solution(x_adams[0])

if len(x_adams) > 1:
    y_adams[1] = rk4_step(x_adams[0], y_adams[0], h_const)
    y_exact_adams[1] = exact_solution(x_adams[1])

for i in range(1, len(x_adams) - 1):
    y_pred, y_corr = adams_pc2_step(x_adams[i-1], y_adams[i-1], x_adams[i], y_adams[i], h_const)
    y_adams[i+1] = y_corr
    y_exact_adams[i+1] = exact_solution(x_adams[i+1])
    

    error_adams_local[i+1] = abs(y_adams[i+1] - y_exact_adams[i+1])
    error_adams_est[i+1] = abs(y_corr - y_pred)

# крок для Адамса
x_adams_auto = [a, a + h_const]
y_adams_auto = [y0, rk4_step(a, y0, h_const)]
h_adams_auto = [h_const, h_const]

curr_x = x_adams_auto[1]
curr_h = h_const
i = 1

while curr_x < b:
    if curr_x + curr_h > b:
        curr_h = b - curr_x
        
    y_pred, y_corr = adams_pc2_step(x_adams_auto[i-1], y_adams_auto[i-1], x_adams_auto[i], y_adams_auto[i], curr_h)
    err = abs(y_corr - y_pred) / 6.0 
    
    if err > eps:
        curr_h /= 2.0
        continue
        
    curr_x += curr_h
    x_adams_auto.append(curr_x)
    y_adams_auto.append(y_corr)
    h_adams_auto.append(curr_h)
    i += 1
    
    if err < eps / 4.0:
        curr_h *= 2.0

# Рунге-Кутта крок 0.01
h_rk4 = 0.01
x_rk4 = np.arange(a, b + h_rk4, h_rk4)
y_rk4 = np.zeros(len(x_rk4))
y_exact_rk4 = np.zeros(len(x_rk4))
error_rk4_local = np.zeros(len(x_rk4))
error_rk4_runge = np.zeros(len(x_rk4))

y_rk4[0] = y0
y_exact_rk4[0] = exact_solution(x_rk4[0])

for i in range(len(x_rk4) - 1):
    y_rk4[i+1] = rk4_step(x_rk4[i], y_rk4[i], h_rk4)
    y_exact_rk4[i+1] = exact_solution(x_rk4[i+1])
    error_rk4_local[i+1] = abs(y_rk4[i+1] - y_exact_rk4[i+1])
    

    y_half_1 = rk4_step(x_rk4[i], y_rk4[i], h_rk4/2)
    y_half_2 = rk4_step(x_rk4[i] + h_rk4/2, y_half_1, h_rk4/2)
    error_rk4_runge[i+1] = (16.0 / 15.0) * abs(y_rk4[i+1] - y_half_2)

# Рунге-Кутта автовибір
x_rk4_auto = [a]
y_rk4_auto = [y0]
h_rk4_auto = [h_const]

curr_x = a
curr_y = y0
curr_h = h_const

while curr_x < b:
    if curr_x + curr_h > b:
        curr_h = b - curr_x
        
    y_full = rk4_step(curr_x, curr_y, curr_h)
    y_half_1 = rk4_step(curr_x, curr_y, curr_h/2)
    y_half_2 = rk4_step(curr_x + curr_h/2, y_half_1, curr_h/2)
    
    err = (16.0 / 15.0) * abs(y_full - y_half_2)
    
    if err > eps:
        curr_h /= 2.0
        continue
        
    curr_x += curr_h
    curr_y = y_full
    x_rk4_auto.append(curr_x)
    y_rk4_auto.append(curr_y)
    h_rk4_auto.append(curr_h)
    
    if err < eps / 32.0:
        curr_h *= 2.0


with open('results_adams.txt', 'w', encoding='utf-8') as f_adams:
    f_adams.write("Метод Адамса (прогноз-корекція 2 порядку)\n")
    f_adams.write("X\t\tY_Adams\t\tY_Exact\t\tLocal_Error\n")
    for i in range(len(x_adams)):
        f_adams.write(f"{x_adams[i]:.4f}\t{y_adams[i]:.6f}\t{y_exact_adams[i]:.6f}\t{error_adams_local[i]:.6e}\n")

with open('results_rk4.txt', 'w', encoding='utf-8') as f_rk4:
    f_rk4.write("Метод Рунге-Кутта 4 порядку\n")
    f_rk4.write("X\t\tY_RK4\t\tY_Exact\t\tLocal_Error\n")
    for i in range(len(x_rk4)):
        f_rk4.write(f"{x_rk4[i]:.4f}\t{y_rk4[i]:.6f}\t{y_exact_rk4[i]:.6f}\t{error_rk4_local[i]:.6e}\n")


plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Адамса 
axs[0, 0].plot(x_adams, error_adams_local, label='Точна похибка', color='blue')
axs[0, 0].plot(x_adams, error_adams_est, label='Оцінка (y_cor - y_pr)', color='red', linestyle='--')
axs[0, 0].set_title('Адамс: Локальна похибка')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('Похибка')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].step(x_adams_auto, h_adams_auto, where='post', color='green')
axs[0, 1].set_title('Адамс: Автоматичний вибір кроку h(x)')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('Крок h')
axs[0, 1].grid(True)

# Рунге-Кутта 
axs[1, 0].plot(x_rk4, error_rk4_local, label='Точна похибка', color='blue')
axs[1, 0].plot(x_rk4, error_rk4_runge, label='Оцінка Рунге', color='red', linestyle='--')
axs[1, 0].set_title('Рунге-Кутта: Локальна похибка')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('Похибка')
axs[1, 0].legend()
axs[1, 0].grid(True)


axs[1, 1].step(x_rk4_auto, h_rk4_auto, where='post', color='green')
axs[1, 1].set_title('Рунге-Кутта: Автоматичний вибір кроку h(x)')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('Крок h')
axs[1, 1].grid(True)



plt.tight_layout()
plt.show()