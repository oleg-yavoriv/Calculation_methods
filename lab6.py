import random
import os

p = os.path.dirname(os.path.abspath(__file__))

def read_matr(name):
    f = open(os.path.join(p, name), 'r')
    m = []
    for line in f:
        row = []
        for x in line.split():
            row.append(float(x))
        m.append(row)
    f.close()
    return m

def read_vect(name):
    f = open(os.path.join(p, name), 'r')
    v = []
    for line in f:
        v.append(float(line))
    f.close()
    return v

def lu_rozklad(A):
    n = len(A)
    L = []
    U = []
    for i in range(n):
        L.append([0] * n)
        U.append([0] * n)
        
    for i in range(n):
        U[i][i] = 1
        
    for k in range(n):
        for i in range(k, n):
            s = 0
            for j in range(k):
                s = s + L[i][j] * U[j][k]
            L[i][k] = A[i][k] - s
            
        for i in range(k + 1, n):
            s = 0
            for j in range(k):
                s = s + L[k][j] * U[j][i]
            U[k][i] = (A[k][i] - s) / L[k][k]
    return L, U

def solve_lu(L, U, B):
    n = len(L)
    Z = []
    X = []
    for i in range(n):
        Z.append(0)
        X.append(0)
        
    for i in range(n):
        s = 0
        for j in range(i):
            s = s + L[i][j] * Z[j]
        Z[i] = (B[i] - s) / L[i][i]
        
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s = s + U[i][j] * X[j]
        X[i] = Z[i] - s
    return X

def mnozhennya(A, X):
    n = len(A)
    res = []
    for i in range(n):
        res.append(0)
        
    for i in range(n):
        s = 0
        for j in range(n):
            s = s + A[i][j] * X[j]
        res[i] = s
    return res

def norma(V):
    m = 0
    for x in V:
        if abs(x) > m:
            m = abs(x)
    return m


n = 100
A = []
for i in range(n):
    row = []
    for j in range(n):
        row.append(random.uniform(1, 10))
    A.append(row)

# записуємо матрицю
f1 = open(os.path.join(p, "matr_A.txt"), "w")
for row in A:
    for x in row:
        f1.write(str(x) + " ")
    f1.write("\n")
f1.close()

x_test = []
for i in range(n):
    x_test.append(2.5)

B = mnozhennya(A, x_test)

# записуємо вектор
f2 = open(os.path.join(p, "vect_B.txt"), "w")
for x in B:
    f2.write(str(x) + "\n")
f2.close()

# читаємо файли назад
A2 = read_matr("matr_A.txt")
B2 = read_vect("vect_B.txt")

L, U = lu_rozklad(A2)

# записуємо LU
f3 = open(os.path.join(p, "LU.txt"), "w")
for row in L:
    for x in row:
        f3.write(str(x) + " ")
    f3.write("\n")
for row in U:
    for x in row:
        f3.write(str(x) + " ")
    f3.write("\n")
f3.close()

X_rozv = solve_lu(L, U, B2)

ax = mnozhennya(A2, X_rozv)
R0 = []
for i in range(n):
    R0.append(B2[i] - ax[i])

print("eps =", norma(R0))

iteraciya = 0
while True:
    iteraciya = iteraciya + 1
    
    ax2 = mnozhennya(A2, X_rozv)
    R = []
    for i in range(n):
        R.append(B2[i] - ax2[i])
        
    dx = solve_lu(L, U, R)
    
    for i in range(n):
        X_rozv[i] = X_rozv[i] + dx[i]
        
    norm_dx = norma(dx)
    norm_r = norma(R)
    
    if norm_dx <= 1e-14 and norm_r <= 1e-14:
        break
        
    if iteraciya > 50:
        break

print("iteraciy =", iteraciya)
print("norma R =", norm_r)
print("norma dx =", norm_dx)