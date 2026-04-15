import numpy as np
import matplotlib.pyplot as plt

t0 = 1.0

def M(t):
    return 50*np.exp(-0.1*t) + 5*np.sin(t)

def dM(t):
    return -5*np.exp(-0.1*t) + 5*np.cos(t)

def diff(t, h):
    return (M(t+h) - M(t-h)) / (2*h)

exact = dM(t0)

h_arr = np.logspace(-20, 3, 1000)
errs = np.abs(diff(t0, h_arr) - exact)
h0 = h_arr[np.nanargmin(errs)]

h = 0.1235
y1 = diff(t0, h)
y2 = diff(t0, 2*h)
y3 = diff(t0, 4*h)

R1 = abs(y1 - exact)
R2 = abs(y2 - exact)

y_R = y1 + (y1 - y2)/3
R3 = abs(y_R - exact)

y_E = (y2**2 - y3*y1) / (2*y2 - (y3+y1))
fin = abs(y_E - exact)
p = (1/np.log(2)) * np.log(abs((y3-y2)/(y2-y1)))

print("R1 =", R1)
print("R2 =", R2)
print("R3 =", R3)
print("Order p =", p)
print("Final error =", fin)
print("Optimal h =", h0)

h_v = np.logspace(-8, -1, 40)
y1_arr = diff(t0, h_v)
y2_arr = diff(t0, 2*h_v)

e1 = np.abs(y1_arr - exact)
e2 = np.abs(y2_arr - exact)
e3 = np.abs((y1_arr + (y1_arr - y2_arr)/3) - exact)

# графік 1
plt.figure()
t_p = np.linspace(0, 20, 200)
plt.plot(t_p, M(t_p))
plt.xlabel("t")
plt.ylabel("M(t)")
plt.grid()
plt.show()

# графік 2
plt.figure()
plt.loglog(h_v, e1, marker='o')
plt.xlabel("h")
plt.ylabel("R1")
plt.grid()
plt.show()

# графік 3
plt.figure()
plt.loglog(h_v, e3, marker='o')
plt.xlabel("h")
plt.ylabel("R3")
plt.grid()
plt.show()

# графік 4
plt.figure()
plt.loglog(h_v, e1, marker='o', label="R1 (h)")
plt.loglog(h_v, e2, marker='s', label="R2 (2h)")
plt.loglog(h_v, e3, marker='^', label="R3 (Richardson)")
plt.xlabel("h")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.show()