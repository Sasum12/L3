#Normal Matrix Multiplication (Single-threaded)
import numpy as np
import time

# Normal matrix multiplication
def matrix_multiply(A, B):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Matrix dimensions do not match!"

    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Example matrices
A = np.random.randint(0, 10, (200, 200))
B = np.random.randint(0, 10, (200, 200))

start = time.time()
C = matrix_multiply(A, B)
end = time.time()

print("Single-threaded time:", round(end - start, 4), "seconds")



#2. Multithreaded Matrix Multiplication (One Thread per Row)

import threading
import numpy as np
import time

def multiply_row(A, B, C, row):
    for j in range(B.shape[1]):
        for k in range(B.shape[0]):
            C[row][j] += A[row][k] * B[k][j]

def threaded_row_multiplication(A, B):
    n = A.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    threads = []

    for i in range(n):
        t = threading.Thread(target=multiply_row, args=(A, B, C, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    return C

# Example
A = np.random.randint(0, 10, (200, 200))
B = np.random.randint(0, 10, (200, 200))

start = time.time()
C = threaded_row_multiplication(A, B)
end = time.time()

print("Multithreaded (one thread per row) time:", round(end - start, 4), "seconds")





#Multithreaded Matrix Multiplication (One Thread per Cell)

def multiply_cell(A, B, C, i, j):
    C[i][j] = sum(A[i][k] * B[k][j] for k in range(A.shape[1]))

def threaded_cell_multiplication(A, B):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Matrix dimensions do not match!"
    C = np.zeros((n, p))
    threads = []

    for i in range(n):
        for j in range(p):
            t = threading.Thread(target=multiply_cell, args=(A, B, C, i, j))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()
    return C

# Example
A = np.random.randint(0, 10, (200, 200))
B = np.random.randint(0, 10, (200, 200))

start = time.time()
C = threaded_cell_multiplication(A, B)
end = time.time()

print("Multithreaded (one thread per cell) time:", round(end - start, 4), "seconds")
