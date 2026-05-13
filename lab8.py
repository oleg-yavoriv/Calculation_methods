import math
import cmath
import matplotlib.pyplot as plt


# функція F(x) = sin(x) - 0.5*x
def f(x):
    return math.sin(x) - 0.5 * x

def df(x):
    return math.cos(x) - 0.5

def d2f(x):
    return -math.sin(x)

def tabulate_and_save(a, b, h, filename):
    x_vals = []
    y_vals = []

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("x\tF(x)\n")
        x = a
        while x <= b + h/10: # h/10 для компенсації похибки float
            y = f(x)
            x_vals.append(x)
            y_vals.append(y)
            file.write(f"{x:.4f}\t{y:.4f}\n")
            x += h
            
    return x_vals, y_vals
    #   графік 1
def plot_function(x_vals, y_vals):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    plt.plot(x_vals, y_vals, color='blue', label='F(x) = sin(x) - 0.5x')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Графік функції F(x)")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.grid(True, color='lightgray')
    plt.legend()
    plt.show()


def simple_iteration(x0, tau, eps):
    x_prev = x0
    iters = 0
    while True:
        # Формула простої релаксації
        x_next = x_prev + tau * f(x_prev)
        iters += 1
        if abs(f(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iters
        x_prev = x_next

def newton(x0, eps):
    x_prev = x0
    iters = 0
    while True:
        # ньютона
        x_next = x_prev - f(x_prev) / df(x_prev)
        iters += 1
        if abs(f(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iters
        x_prev = x_next

def chebyshev(x0, eps):
    x_prev = x0
    iters = 0
    while True:
        #  Чебишев
        x_next = x_prev - f(x_prev)/df(x_prev) - 0.5 * (f(x_prev)**2 * d2f(x_prev)) / (df(x_prev)**3)
        iters += 1
        if abs(f(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iters
        x_prev = x_next

def secant(x0, x1, eps):
    x_prev = x1
    x_pprev = x0
    iters = 0
    while True:
        # хорд
        x_next = x_prev - f(x_prev) * (x_prev - x_pprev) / (f(x_prev) - f(x_pprev))
        iters += 1
        if abs(f(x_next)) < eps and abs(x_next - x_prev) < eps:
            return x_next, iters
        x_pprev = x_prev
        x_prev = x_next

# парабол
def div_diff1(x0, x1): 
    return (f(x1) - f(x0)) / (x1 - x0)
def div_diff2(x0, x1, x2): 
    return (div_diff1(x1, x2) - div_diff1(x0, x1)) / (x2 - x0)

def parabola(x0, x1, x2, eps):
    xn2, xn1, xn = x0, x1, x2
    iters = 0
    while True:
        A = div_diff2(xn2, xn1, xn)
        B = (xn - xn1) * A + div_diff1(xn1, xn)
        C = f(xn)
        
        D = B**2 - 4*A*C
        sqrt_D = cmath.sqrt(D)
        
        delta1 = (-B + sqrt_D) / (2*A)
        delta2 = (-B - sqrt_D) / (2*A)
        
        if abs(delta1) < abs(delta2):
            delta = delta1
        else:
            delta = delta2
            
        x_next = xn + delta.real
        iters += 1
        
        if abs(f(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iters
        
        xn2, xn1, xn = xn1, xn, x_next

def inverse_interp(x0, x1, x2, eps):
    xn2, xn1, xn = x0, x1, x2
    iters = 0
    while True:
        y2, y1, y0 = f(xn2), f(xn1), f(xn)
        # Лагранжа
        t1 = (y1 * y0) / ((y2 - y1) * (y2 - y0)) * xn2
        t2 = (y2 * y0) / ((y1 - y2) * (y1 - y0)) * xn1
        t3 = (y2 * y1) / ((y0 - y2) * (y0 - y1)) * xn
        
        x_next = t1 + t2 + t3
        iters += 1
        
        if abs(f(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iters
        
        xn2, xn1, xn = xn1, xn, x_next





def save_poly_coeffs(filename, coeffs):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(" ".join(map(str, coeffs)))

def read_poly_coeffs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readline()
        return [float(c) for c in line.split()]

# Горнера
def horner(coeffs, x):
    m = len(coeffs) - 1
    b = [0] * (m + 1)
    b[m] = coeffs[m]
    for i in range(m - 1, -1, -1):
        b[i] = coeffs[i] + x * b[i+1]
    return b[0], b

def newton_horner(coeffs, x0, eps):
    x = x0
    iters = 0
    while True:
        # Ньютона по Горнера
        px, b = horner(coeffs, x)
        dfx, _ = horner(b[1:], x)
        
        x_next = x - px / dfx
        iters += 1
        if abs(px) < eps and abs(x_next - x) < eps:
            return x_next, iters
        x = x_next

def lin_method(coeffs, p0, q0, eps):
    a0, a1, a2, a3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    p = p0
    q = q0
    iters = 0
    while True:
        # Ліна
        b3 = a3
        b2 = a2 - p * b3
        
        q1 = a0 / b2
        p1 = (a1 * b2 - a0 * b3) / (b2**2)
        
        alpha1 = -p1 / 2
        beta1 = cmath.sqrt(q1 - alpha1**2).real
        
        alpha0 = -p / 2
        beta0 = cmath.sqrt(q - alpha0**2).real
        
        iters += 1
        if abs(alpha1 - alpha0) < eps and abs(beta1 - beta0) < eps:
            return complex(alpha1, beta1), complex(alpha1, -beta1), iters
        
        p = p1
        q = q1
    #   графік 2 
def plot_poly(coeffs, a, b):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    
    x_vals = []
    y_vals = []
    x = a
    while x <= b:
        y, _ = horner(coeffs, x)
        x_vals.append(x)
        y_vals.append(y)
        x += 0.1
        
    plt.plot(x_vals, y_vals, color='green', label='P(x) = x^3 - 2x^2 + x - 2')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Графік алгебраїчного рівняння")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.grid(True, color='lightgray')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    eps = 1e-10
    
    print("=== Завдання 1-4: Нелінійне рівняння ===")
    x_vals, y_vals = tabulate_and_save(-3, 3, 0.1, "tabulation.txt")
    print("Табуляцію записано у файл 'tabulation.txt'.")
    plot_function(x_vals, y_vals)
    
    x0_inc = 0.5
    x0_dec = 2.0
    
    print("\n-- Корінь 1 (зростання) --")
    r, it = simple_iteration(x0_inc, -0.9, eps) # tau підбираємо для збіжності
    print(f"Проста ітерація: x = {r:.8f}, кроків = {it}")
    r, it = newton(x0_inc, eps)
    print(f"Метод Ньютона:   x = {r:.8f}, кроків = {it}")
    r, it = chebyshev(x0_inc, eps)
    print(f"Метод Чебишева:  x = {r:.8f}, кроків = {it}")
    r, it = secant(0.0, x0_inc, eps)
    print(f"Метод хорд:      x = {r:.8f}, кроків = {it}")
    r, it = parabola(0.0, 0.2, x0_inc, eps)
    print(f"Метод парабол:   x = {r:.8f}, кроків = {it}")
    r, it = inverse_interp(0.0, 0.2, x0_inc, eps)
    print(f"Зворотна інтерп: x = {r:.8f}, кроків = {it}")

    print("\n-- Корінь 2 (спадання) --")
    r, it = simple_iteration(x0_dec, 0.9, eps) 
    print(f"Проста ітерація: x = {r:.8f}, кроків = {it}")
    r, it = newton(x0_dec, eps)
    print(f"Метод Ньютона:   x = {r:.8f}, кроків = {it}")
    r, it = chebyshev(x0_dec, eps)
    print(f"Метод Чебишева:  x = {r:.8f}, кроків = {it}")
    r, it = secant(1.5, x0_dec, eps)
    print(f"Метод хорд:      x = {r:.8f}, кроків = {it}")
    r, it = parabola(1.5, 1.7, x0_dec, eps)
    print(f"Метод парабол:   x = {r:.8f}, кроків = {it}")
    r, it = inverse_interp(1.5, 1.7, x0_dec, eps)
    print(f"Зворотна інтерп: x = {r:.8f}, кроків = {it}")


    print("\n=== Завдання 5-9: Алгебраїчне рівняння ===")
    my_coeffs = [-2, 1, -5, 1] 
    
    save_poly_coeffs("poly_coeffs.txt", my_coeffs)
    print("Коефіцієнти записано у файл 'poly_coeffs.txt'.")
    loaded_coeffs = read_poly_coeffs("poly_coeffs.txt")
    
    plot_poly(loaded_coeffs, -1, 3)
    
    r_real, it_r = newton_horner(loaded_coeffs, 2.5, eps)
    print(f"Дійсний корінь (Ньютон-Горнер): x = {r_real:.8f}, кроків = {it_r}")
    
    c1, c2, it_c = lin_method(loaded_coeffs, 0.5, 0.5, eps)
    print(f"Комплексні корені (метод Ліна): x1 = {c1:.8f}, x2 = {c2:.8f}, кроків = {it_c}")