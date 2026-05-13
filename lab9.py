import matplotlib.pyplot as plt


# Функція Розенброка 
def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

def phi_func(x):
    f1 = x[0]**2 + x[1] - 2
    f2 = x[0] - x[1]
    return f1**2 + f2**2


def plot_equations():
    x_vals = []
    y1_vals = []
    y2_vals = []
    

    x_temp = -3.0
    while x_temp <= 3.0:
        x_vals.append(x_temp)
        # З f1=0: x2 = 2 - x1^2
        y1_vals.append(2 - x_temp**2)
        # З f2=0: x2 = x1
        y2_vals.append(x_temp)
        x_temp += 0.1

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y1_vals, label="x1^2 + x2 - 2 = 0", color="blue")
    plt.plot(x_vals, y2_vals, label="x1 - x2 = 0", color="red")
    plt.title("Графіки системи рівнянь")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()
    plt.show()

def hooke_jeeves(func, x0, step, q, p, eps1, filename):
    x_base = list(x0)
    n = len(x0)
    steps_count = 0
    
    #  ФАЙЛ
    file = open(filename, "w")
    file.write("Початок пошуку методом Хука-Дживса\n")
    
    while True:
        steps_count += 1
        file.write(f"Крок {steps_count}: Координати {x_base}, Значення функції = {func(x_base)}\n")
        
        x_explore = list(x_base)
    
        for i in range(n):
            temp = list(x_explore)
            temp[i] += step[i]
            if func(temp) < func(x_explore):
                x_explore = list(temp)
            else:
                temp[i] -= 2 * step[i]
                if func(temp) < func(x_explore):
                    x_explore = list(temp)
                    
        if x_explore != x_base:
            x_pattern = []
            for i in range(n):
                x_pattern.append(x_explore[i] + p * (x_explore[i] - x_base[i]))
                
            x_explore_pattern = list(x_pattern)
            
            for i in range(n):
                temp = list(x_explore_pattern)
                temp[i] += step[i]
                if func(temp) < func(x_explore_pattern):
                    x_explore_pattern = list(temp)
                else:
                    temp[i] -= 2 * step[i]
                    if func(temp) < func(x_explore_pattern):
                        x_explore_pattern = list(temp)
                        
            if func(x_explore_pattern) < func(x_explore):
                x_base = list(x_explore)
                x_explore = list(x_explore_pattern)
            else:
                x_base = list(x_explore)
        else:
            for i in range(n):
                step[i] /= q
                
        max_step = max(step)
        if max_step < eps1:
            break
            

    file.write(f"\nПОШУК ЗАВЕРШЕНО ЗА {steps_count} КРОКІВ.\n")
    file.write(f"Точка мінімуму: {x_base}\n")
    file.close()
    
    return x_base, steps_count



if __name__ == "__main__":
    # графік 
    plot_equations()
    
    # Розенброка
    print("Тестування на функції Розенброка...")
    x0_test = [-1.2, 0.0]
    step_test = [0.5, 0.5]
    res_test, steps_test = hooke_jeeves(rosenbrock, x0_test, step_test, q=2, p=2, eps1=0.001, filename="rosenbrock_trajectory.txt")
    print(f"Результат тестування: {res_test} за {steps_test} кроків. Дані збережено у 'rosenbrock_trajectory.txt'\n")
    
    print("Розв'язок системи нелінійних рівнянь (пошук мінімуму Phi)...")
    x0_sys = [0.5, 0.5] # Початкове наближення
    step_sys = [0.5, 0.5]
    res_sys, steps_sys = hooke_jeeves(phi_func, x0_sys, step_sys, q=2, p=2, eps1=0.001, filename="system_trajectory.txt")
    print(f"Знайдений розв'язок системи: {res_sys} за {steps_sys} кроків. Дані збережено у 'system_trajectory.txt'")