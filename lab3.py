import numpy as np
import matplotlib.pyplot as plt
import csv

def prepare_and_read_data(filename="temp_data.csv"):
    x_raw = np.arange(1, 25)
    y_raw = [-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0, 
             -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Month', 'Temp'])
        for x, y in zip(x_raw, y_raw):
            writer.writerow([x, y])

    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
            
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)

def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1), dtype=np.float64)
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x**(i + j))
    return A

def form_vector(x, y, m):
    b = np.zeros(m + 1, dtype=np.float64)
    for i in range(m + 1):
        b[i] = np.sum(y * (x**i))
    return b

def gauss_solve(A, b):
    n = len(b)
    A, b = A.copy(), b.copy()
    
    for k in range(n - 1):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if max_row != k:
            A[[k, max_row]], b[[k, max_row]] = A[[max_row, k]], b[[max_row, k]]
            
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
            
    x_sol = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i+1:], x_sol[i+1:])) / A[i, i]
        
    return x_sol

def polynomial(x, coef):
    y_poly = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(coef):
        y_poly += c * (x**i)
    return y_poly

def variance(y_true, y_approx, m):
    n = len(y_true)
    if n - m - 1 <= 0:
        return float('inf') 
    return np.sum((y_true - y_approx)**2) / (n - m - 1)

def run_analysis():
    x, y = prepare_and_read_data()
    n_nodes = len(x)
    max_degree = 10
    variances = []
    
    print("--- Аналіз дисперсії ---")
    for m in range(1, max_degree + 1):
        coef = gauss_solve(form_matrix(x, m), form_vector(x, y, m))
        var = variance(y, polynomial(x, coef), m)
        variances.append(var)
        print(f"Степінь m={m}: Дисперсія = {var:.4f}")

    optimal_m = np.argmin(variances) -5
    print(f"\n=> Оптимальний степінь многочлена: m = {optimal_m}")

    coef_opt = gauss_solve(form_matrix(x, optimal_m), form_vector(x, y, optimal_m))
    
    x_future = np.array([25, 26, 27], dtype=np.float64)
    y_future = polynomial(x_future, coef_opt)
    
    print("\n--- Прогноз на наступні 3 місяці ---")
    for mth, temp in zip(x_future, y_future):
        print(f"Місяць {int(mth)}: {temp:.2f} °C")

    h1 = (x[-1] - x[0]) / (20 * n_nodes)
    x_fine = np.arange(x[0], x[-1], h1)
    y_fine_poly = polynomial(x_fine, coef_opt)
    error_fine = np.abs(np.interp(x_fine, x, y) - y_fine_poly)
    residuals = y - polynomial(x, coef_opt)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].plot(range(1, max_degree + 1), variances, 'o-', color='purple')
    axes[0].axvline(optimal_m, color='red', linestyle='--', label=f'Опт. m={optimal_m}')
    axes[0].set(title='Залежність дисперсії від степеня m', xlabel='m', ylabel='Дисперсія')
    axes[0].grid(True); axes[0].legend()

    axes[1].plot(x_fine, y_fine_poly, 'b-', label=f'Апроксимація (m={optimal_m})')
    axes[1].plot(x, y, 'ko', label='Фактичні дані')
    axes[1].plot(x_future, y_future, 'r*', markersize=10, label='Прогноз')
    axes[1].set(title='Апроксимація температур (МНК)', xlabel='Місяць', ylabel='Температура (°C)')
    axes[1].grid(True); axes[1].legend()

    axes[2].plot(x_fine, error_fine, 'r-')
    axes[2].set(title=f'Похибка апроксимації (Крок h={h1:.4f})', xlabel='Місяць', ylabel='Абс. похибка')
    axes[2].grid(True)

    axes[3].bar(x, residuals, color='orange', alpha=0.7, edgecolor='black')
    axes[3].axhline(0, color='black', linestyle='--')
    axes[3].set(title=f'Графік залишків (m={optimal_m})', xlabel='Місяць', ylabel='Відхилення від моделі (°C)')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_analysis()
