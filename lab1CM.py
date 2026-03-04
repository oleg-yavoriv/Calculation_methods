import numpy as np
import matplotlib.pyplot as plt
import requests

def get_route_data():
    # Пункт 1-2: Запит до Open-Elevation API
    url = (
        "https://api.open-elevation.com/api/v1/lookup?locations="
        "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
        "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
        "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
        "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
        "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
        "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
        "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
    )
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    return response.json()['results']

def haversine(lat1, lon1, lat2, lon2):
    # Пункт 4: Обчислення кумулятивної відстані
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

class CubicSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.calculate_coefficients()

    def calculate_coefficients(self):
        # Пункт 6: Коефіцієнти системи лінійних рівнянь
        n = self.n - 1
        h = np.diff(self.x)
        self.a = self.y[:-1] 
        
        # Діагоналі матриці (A - нижня, B - головна, C - верхня, F - вільні члени)
        self.A_diag = h[:-1]
        self.B_diag = 2 * (h[:-1] + h[1:])
        self.C_diag = h[1:]
        self.F_vec = 3 * ((self.y[2:] - self.y[1:-1]) / h[1:] - (self.y[1:-1] - self.y[:-2]) / h[:-1])
        
        self.alpha_sweep = np.zeros(n-1)
        self.beta_sweep = np.zeros(n-1)
        
        if n > 1:
            self.alpha_sweep[0] = -self.C_diag[0] / self.B_diag[0]
            self.beta_sweep[0] = self.F_vec[0] / self.B_diag[0]
            
            # Пряма прогонка
            for i in range(1, n - 1):
                denom = self.B_diag[i] + self.A_diag[i] * self.alpha_sweep[i-1]
                self.alpha_sweep[i] = -self.C_diag[i] / denom
                self.beta_sweep[i] = (self.F_vec[i] - self.A_diag[i] * self.beta_sweep[i-1]) / denom
                
            # Зворотна прогонка
            c_int = np.zeros(n-1)
            c_int[-1] = (self.F_vec[-1] - self.A_diag[-1] * self.beta_sweep[-2]) / (self.B_diag[-1] + self.A_diag[-1] * self.alpha_sweep[-2])
            for i in range(n-3, -1, -1):
                c_int[i] = self.alpha_sweep[i] * c_int[i+1] + self.beta_sweep[i]
        else:
            c_int = []

        self.c = np.zeros(self.n)
        if len(c_int) > 0:
            self.c[1:-1] = c_int
            
        self.d = (self.c[1:] - self.c[:-1]) / (3 * h)
        self.b = (self.y[1:] - self.y[:-1]) / h - (h * (self.c[1:] + 2 * self.c[:-1])) / 3

    def interpolate(self, x_val):
        if x_val <= self.x[0]: return self.y[0]
        if x_val >= self.x[-1]: return self.y[-1]
        
        idx = np.searchsorted(self.x, x_val) - 1
        dx = x_val - self.x[idx]
        return self.a[idx] + self.b[idx]*dx + self.c[idx]*(dx**2) + self.d[idx]*(dx**3)

def main():
    results = get_route_data()
    n_points = len(results)
    
    lats = [p['latitude'] for p in results]
    lons = [p['longitude'] for p in results]
    elevs = [p['elevation'] for p in results]
    
    distances = [0.0]
    for i in range(1, n_points):
        d = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
        distances.append(distances[-1] + d)
    
    x_full = np.array(distances)
    y_full = np.array(elevs)

    # Пункт 3: Запис у текстовий файл
    with open("lab1_results.txt", "w", encoding="utf-8") as f:
        f.write("N | Latitude | Longitude | Elevation\n")
        f.write("-" * 45 + "\n")
        for i in range(n_points):
            f.write(f"{i:<3} | {lats[i]:<10.6f} | {lons[i]:<10.6f} | {elevs[i]:<9.2f}\n")
        f.write("\nТабуляція (відстань, висота)\n")
        for i in range(n_points):
            f.write(f"{i:<3} | Distance: {distances[i]:.2f} m | Elev: {elevs[i]:.2f} m\n")
    print("Дані успішно збережено у файл 'lab1_results.txt'")

    # Вивід результатів у консоль
    print(f"Кількість вузлів: {n_points}")
    print("\nТабуляція вузлів:")
    print(f"{'N':<3} | {'Latitude':<10} | {'Longitude':<10} | {'Elevation':<9}")
    for i in range(n_points):
        print(f"{i:<3} | {lats[i]:<10.6f} | {lons[i]:<10.6f} | {elevs[i]:<9.2f}")

    print("\nТабуляція (відстань, висота)")
    for i in range(n_points):
        print(f"{i:<3} | {distances[i]:<12.2f} | {elevs[i]:<9.2f}")

    spline_etalon = CubicSpline(x_full, y_full)
    
    # Пункт 7: Вивід коефіцієнтів методу прогонки (alpha, beta)
    print("\nКоефіцієнти методу прогонки (alpha, beta):")
    print(f"{'i':<3} | {'alpha_i':<12} | {'beta_i':<12}")
    for i in range(len(spline_etalon.alpha_sweep)):
        print(f"{i+1:<3} | {spline_etalon.alpha_sweep[i]:<12.4f} | {spline_etalon.beta_sweep[i]:<12.4f}")

    # Пункт 8-9: ВИВІД КОЕФІЦІЄНТІВ СПЛАЙНУ
    print("\nТаблиця коефіцієнтів кубічного сплайна (a_i, b_i, c_i, d_i):")
    print(f"{'i':<3} | {'a_i':<10} | {'b_i':<10} | {'c_i':<12} | {'d_i':<12}")
    for i in range(spline_etalon.n - 1):
        print(f"{i+1:<3} | {spline_etalon.a[i]:<10.2f} | {spline_etalon.b[i]:<10.4f} | {spline_etalon.c[i]:<12.6f} | {spline_etalon.d[i]:<12.8f}")

    # ДОДАТКОВІ ЗАВДАННЯ (Маршрут, Енергія, Градієнт)
    total_ascent = sum(max(elevs[i] - elevs[i-1], 0) for i in range(1, n_points))
    total_descent = sum(max(elevs[i-1] - elevs[i], 0) for i in range(1, n_points))
    print(f"\nЗагальна довжина маршруту (м): {distances[-1]:.2f}")
    print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
    print(f"Сумарний спуск (м): {total_descent:.2f}")
    
    mass = 80
    g = 9.81
    energy = mass * g * total_ascent
    print(f"Механічна робота (Дж): {energy:.2f}")
    print(f"Механічна робота (кДж): {energy/1000:.2f}")
    print(f"Енергія (ккал): {energy / 4184:.2f}")

    grad_full = np.gradient(y_full, x_full) * 100
    print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
    print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
    print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

    print("\n==================================")

    x_dense = np.linspace(x_full[0], x_full[-1], 500)
    y_etalon_dense = np.array([spline_etalon.interpolate(xi) for xi in x_dense])

    # Пункт 5, 10, 12: Побудова графіків
    fig0, ax0 = plt.subplots(figsize=(10, 5))
    ax0.set_title("Профіль висот маршруту (Заросляк -> Говерла)")
    ax0.plot(x_full, y_full, marker='o', color='teal', label='Висота маршруту')
    ax0.fill_between(x_full, y_full, min(y_full) - 50, color='teal', alpha=0.2)
    ax0.set_xlabel("Кумулятивна відстань (м)")
    ax0.set_ylabel("Висота над рівнем моря (м)")
    ax0.grid(True, linestyle='--', alpha=0.6)
    ax0.legend()
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_title("Вплив кількості вузлів")
    ax1.plot(x_dense, y_etalon_dense, label='21 вузол (еталон)')

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_title("Похибка апроксимації")

    for count in [10, 15, 20]:
        indices = np.linspace(0, n_points - 1, count, dtype=int)
        spline = CubicSpline(x_full[indices], y_full[indices])
        y_dense_s = np.array([spline.interpolate(xi) for xi in x_dense])
        
        ax1.plot(x_dense, y_dense_s, label=f'{count} вузлів')
        errors = np.abs(y_dense_s - y_etalon_dense)
        ax2.plot(x_dense, errors, label=f'{count} вузлів')
        
        print(f"\n{count} вузлів")
        print(f"Максимальна похибка: {np.max(errors)}")
        print(f"Середня похибка: {np.mean(errors)}")

    ax1.legend()
    ax2.legend()
    plt.show()

if __name__ == "__main__": 
    main()