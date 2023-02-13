import numpy as np
import ctypes


def HouseHolder_Copy_Row(A,k):
    v=np.copy(A[k:,k])
    mu=v[0]+(v[0]/abs(v[0]))*np.linalg.norm(v,2)
    if abs(mu)>=1e-6:
        v[1:]/=mu
    v[0]=1.0
    return v

def HouseHolder_Copy_Col(A,k):
    v=np.copy(A[k,k+1:])
    mu=v[0]+(v[0]/abs(v[0]))*np.linalg.norm(v,2)
    if abs(mu)>=1e-6:
        v[1:]/=mu
    v[0]=1.0
    return v

def HouseHolder_Row(A,v,k):
    beta=-2/np.dot(v,v)
    A_T=np.copy(A[k:,k:])
    w=beta*np.matmul(np.transpose(A_T),v)
    out=np.outer(v,w)
    return out

def HouseHolder_Col(A,v,k):
    beta=-2/np.dot(v,v)
    w=beta*np.matmul(A[k:,k+1:],v)
    out=np.outer(w,v)
    return out


Mat=np.ndarray([[0.59828957, 0.29664485, 0.9068596,  0.80018892, 0.12259314, 0.34763641],
 [0.90453219, 0.73585238, 0.16374527, 0.12865663, 0.06864848, 0.50134387],
 [0.31981082, 0.77745353, 0.41087337, 0.96505045, 0.82118556, 0.22923782],
 [0.40948946, 0.10791781, 0.32800317, 0.7647702,  0.74662561, 0.28577958],
 [0.36770176, 0.12763882, 0.73809877, 0.76038764, 0.47508413, 0.17154537],
 [0.79979308, 0.93534805, 0.62592693, 0.65035382, 0.80558099, 0.18553789],
 [0.94381072, 0.27746098, 0.78284145, 0.70678616, 0.37961537, 0.84736446]])
col=np.shape(Mat)[1]
row=np.shape(Mat)[0]
print("Original Matrix\n", Mat)


for i in range(col):
    v=HouseHolder_Copy_Row(Mat,i)
    outer=HouseHolder_Row(Mat,v,i)
    Mat[i:,i:]+=outer
    if(i<=col-2):
        v=HouseHolder_Copy_Col(Mat,i)
        outer=HouseHolder_Col(Mat,v,i)
        Mat[i:,i+1:]+=outer
print("Final Matrix\n", Mat)