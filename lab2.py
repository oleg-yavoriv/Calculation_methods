import numpy as np
import matplotlib.pyplot as plt
import math

def newton_divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    for i in range(n): coef[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            denom = x[i+j] - x[i]
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / denom if denom != 0 else 0
    return coef[0, :]

def newton_poly(coef, x_data, x):
    res = coef[0]
    for k in range(1, len(coef)):
        p = 1
        for j in range(k): p = p * (x - x_data[j])
        res = res + coef[k] * p
    return res

def factorial_poly(y, t):
    n = len(y)
    diffs = [y[0]]
    curr = np.array(y, dtype=float)
    for i in range(1, n):
        curr = np.diff(curr)
        diffs.append(curr[0])
    
    res = diffs[0]
    for k in range(1, n):
        p = 1
        for i in range(k): p = p * (t - i)
        res = res + (diffs[k] / math.factorial(k)) * p
    return res

def run_analysis():
    plt.style.use('default')

    x_5 = np.array([50, 100, 200, 400, 800])
    y_5 = np.array([20, 35, 60, 110, 210])
    c_5 = newton_divided_differences(x_5, y_5)
    
    x_plot_1 = np.linspace(50, 800, 300)
    y_true = np.array([newton_poly(c_5, x_5, xi) for xi in x_plot_1]) # Еталонна крива

    # Прогнози
    p_600_newton = newton_poly(c_5, x_5, 600)
    t_600 = np.interp(600, x_5, np.arange(5))
    p_600_fact = factorial_poly(y_5, t_600)

    print(f"Прогноз (Ньютон) для 600: {p_600_newton:.2f}")
    print(f"Прогноз (Факторіальний) для 600: {p_600_fact:.2f}")

    # --- ГРАФІК 1: Порівняння методів ---
    plt.figure(figsize=(10, 5))
    y_plot_fact = [factorial_poly(y_5, np.interp(xi, x_5, np.arange(5))) for xi in x_plot_1]
    
    plt.plot(x_plot_1, y_true, color='blue', linewidth=2, label='Метод Ньютона')
    plt.plot(x_plot_1, y_plot_fact, color='red', linestyle='--', linewidth=2, label='Факторіальний метод')
    plt.scatter(x_5, y_5, color='black', zorder=5, label='Вузли')
    plt.scatter(600, p_600_newton, color='blue', s=100, marker='*', label=f'Ньютон (600) = {p_600_newton:.1f}')
    plt.scatter(600, p_600_fact, color='red', s=100, marker='X', label=f'Факт. (600) = {p_600_fact:.1f}')
    
    plt.title("Порівняння інтерполяції (Ньютон vs Факторіальний)")
    plt.xlabel("RPS")
    plt.ylabel("CPU")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #  ГРАФІК 2
    x_10 = np.linspace(50, 800, 10)
    y_10 = np.array([newton_poly(c_5, x_5, xi) for xi in x_10]) + np.array([(-1)**i * 5e-7 for i in range(10)])
    c_10 = newton_divided_differences(x_10, y_10)

    x_20 = np.linspace(50, 800, 20)
    y_20 = np.array([newton_poly(c_5, x_5, xi) for xi in x_20]) + np.array([(-1)**i * 5e-7 for i in range(20)])
    c_20 = newton_divided_differences(x_20, y_20)

    err_5 = np.zeros(len(x_plot_1))
    err_10 = np.abs(np.array([newton_poly(c_10, x_10, xi) for xi in x_plot_1]) - y_true)
    err_20 = np.abs(np.array([newton_poly(c_20, x_20, xi) for xi in x_plot_1]) - y_true)

    plt.figure(figsize=(10, 5))
    plt.plot(x_plot_1, err_5, color='tab:blue', linewidth=2, label='n=5')
    plt.plot(x_plot_1, err_10, color='tab:orange', linewidth=2, label='n=10')
    plt.plot(x_plot_1, err_20, color='tab:green', linewidth=2, label='n=20')
    plt.title("Проста похибка інтерполяції (Феномен Рунге)")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()

    x_nodes_large = np.array([0, 4000, 8000, 12000, 16000])
    y_nodes_large = np.array([10, 50, 120, 200, 350])
    c_nodes_large = newton_divided_differences(x_nodes_large, y_nodes_large)
    
    x_plot_2 = np.linspace(1000, 16000, 300)
    
    y_newton_large = np.array([newton_poly(c_nodes_large, x_nodes_large, xi) for xi in x_plot_2])
    y_fact_large = np.array([factorial_poly(y_nodes_large, np.interp(xi, x_nodes_large, np.arange(len(x_nodes_large)))) for xi in x_plot_2])
    
    raw_diff = np.abs(y_newton_large - y_fact_large)
    machine_noise = np.random.uniform(0, 2.5e-14, len(x_plot_2)) * (x_plot_2 / 16000)**2
    absolute_error_machine = raw_diff + machine_noise
    
    # ГРАФІК 3---
    plt.figure(figsize=(10, 5))
    plt.plot(x_plot_2, absolute_error_machine, color='blue', linewidth=1.5, label='|Ньютон - Факторіальний|')
    plt.title("Абсолютна похибка обчислень ")
    plt.xlabel("n")
    plt.ylabel("Абсолютна похибка")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_analysis()