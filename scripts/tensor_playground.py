import numpy as np

A = np.zeros((4, 3))
for i in range(4):
    for j in range(3):
        A[i, j] = i * 10 + j

T = np.zeros((5, 4, 3))
for k in range(5):
    for i in range(4):
        for j in range(3):
            T[k, i, j] = k * 100 + i * 10 + j

D = np.zeros((6, 5, 4, 3))
for n in range(6):
    for k in range(5):
        for i in range(4):
            for j in range(3):
                D[n, k, i, j] = n * 1000 + k * 100 + i * 10 + j

print(A)

print(T)

print(D)
