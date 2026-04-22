import random

def generate_system(n=100):
    A = []
    for i in range(n):
        row = []
        sum_off_diag = 0
        for j in range(n):
            if i != j:
                val = random.uniform(-10.0, 10.0)
                row.append(val)
                sum_off_diag += abs(val)
            else:
                row.append(0.0) 
        
        row[i] = sum_off_diag + random.uniform(1.0, 10.0) 
        A.append(row)     
    # x_i = 2.5)
    X_true = [2.5] * n
    B = []
    for i in range(n):
        b_i = sum(A[i][j] * X_true[j] for j in range(n))
        B.append(b_i)
        
    return A, B

def save_to_file(filename, data, is_matrix=False):
    with open(filename, 'w') as f:
        if is_matrix:
            for row in data:
                f.write("\t".join(map(str, row)) + "\n")
        else:
            for item in data:
                f.write(str(item) + "\n")

def read_matrix(filename):
    A = []
    with open(filename, 'r') as f:
        for line in f:
            A.append([float(x) for x in line.split()])
    return A

def read_vector(filename):
    B = []
    with open(filename, 'r') as f:
        for line in f:
            B.append(float(line.strip()))
    return B

def mat_vec_mult(A, X):
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        result[i] = sum(A[i][j] * X[j] for j in range(n))
    return result

def vec_norm(X):
    return max(abs(x) for x in X)

def vec_diff(X1, X2):
    return [x1 - x2 for x1, x2 in zip(X1, X2)]

def mat_norm(A):
    return max(sum(abs(x) for x in row) for row in A)

def method_simple_iteration(A, B, eps=1e-14):
    n = len(A)
    X = [1.0] * n  # Початкове наближення 
    tau = 1.0 / mat_norm(A) # Умова збіжності 
    
    iters = 0
    while True:
        AX = mat_vec_mult(A, X)
        X_new = [0.0] * n
        for i in range(n):
            # X^(k+1) = X^(k) - tau * (A*X^(k) - B)
            X_new[i] = X[i] - tau * (AX[i] - B[i])
        if vec_norm(vec_diff(X_new, X)) < eps:
            return X_new, iters + 1
            
        X = X_new
        iters += 1
        
        if iters > 100000: 
            break
    return X, iters

def method_jacobi(A, B, eps=1e-14):
    n = len(A)
    X = [1.0] * n # Початкове наближення
    
    iters = 0
    while True:
        X_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * X[j] for j in range(n) if i != j)
            X_new[i] = (B[i] - s) / A[i][i]
            
        if vec_norm(vec_diff(X_new, X)) < eps:
            return X_new, iters + 1
            
        X = X_new
        iters += 1
    return X, iters

def method_seidel(A, B, eps=1e-14):
    n = len(A)
    X = [1.0] * n # Початкове наближення
    
    iters = 0
    while True:
        X_old = X.copy()
        for i in range(n):
            s = sum(A[i][j] * X[j] for j in range(n) if i != j)
            X[i] = (B[i] - s) / A[i][i]
            
        if vec_norm(vec_diff(X, X_old)) < eps:
            return X, iters + 1
            
        iters += 1
    return X, iters

if __name__ == "__main__":
    n_size = 100
    epsilon = 1e-14

    A_gen, B_gen = generate_system(n_size)
    save_to_file("matrix_A.txt", A_gen, is_matrix=True)
    save_to_file("vector_B.txt", B_gen)
    
    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")
    
    # Метод простої ітерації
    X_simp, iters_simp = method_simple_iteration(A, B, eps=epsilon)
    print(f"1. Метод простої ітерації:")
    print(f"   Ітерацій: {iters_simp}")
    print(f"   x[0] = {X_simp[0]:.6f}, x[-1] = {X_simp[-1]:.6f}")
    
    # Метод Якобі
    X_jacobi, iters_jacobi = method_jacobi(A, B, eps=epsilon)
    print(f"\n2. Метод Якобі:")
    print(f"   Ітерацій: {iters_jacobi}")
    print(f"   x[0] = {X_jacobi[0]:.6f}, x[-1] = {X_jacobi[-1]:.6f}")
    
    # Метод Зейделя
    X_seidel, iters_seidel = method_seidel(A, B, eps=epsilon)
    print(f"\n3. Метод Зейделя:")
    print(f"   Ітерацій: {iters_seidel}")
    print(f"   x[0] = {X_seidel[0]:.6f}, x[-1] = {X_seidel[-1]:.6f}")