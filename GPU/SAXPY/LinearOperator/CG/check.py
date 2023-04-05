import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse.linalg import cg

row=np.array([9.71338,
0.394383,
0.783099,
0.79844,
0.911647,
0.197551,
0.335223,
0.76823,
0.277775,
0.55397,
0.477397,
0.628871,
0.364784,
0.513401,
0.95223,
0.916195])

x=np.array([
0.61264,
0.296032,
0.637552,
0.524287,
0.493583,
0.972775,
0.292517,
0.771358,
0.526745,
0.769914,
0.400229,
0.891529,
0.283315,
0.352458,
0.807725,
0.919026,
])

b=np.array([
0.635712,
0.717297,
0.141603,
0.606969,
0.0163006,
0.242887,
0.137232,
0.804177,
0.156679,
0.400944,
0.12979,
0.108809,
0.998924,
0.218257,
0.512932,
0.839112,
])
A=toeplitz(c=row,r=row)




x_fin=cg(A=A,b=b,x0=x)
print(x_fin)