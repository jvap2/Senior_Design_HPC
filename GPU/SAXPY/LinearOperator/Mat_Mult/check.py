import numpy as np
from scipy.linalg import toeplitz

r_A=np.array([
0.933273,
0.786414,
0.628502,
0.286094,
])

r_B=np.array([
0.0679756,
0.177357,
0.0530033,
0.968207,
])

A=toeplitz(r=r_A, c=r_A)
B=toeplitz(r=r_B, c=r_B)


C=np.matmul(A,B)
print(C)