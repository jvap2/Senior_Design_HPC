import numpy as np


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


Mat=np.random.rand(4,3)
col=np.shape(Mat)[1]
row=np.shape(Mat)[0]
print(Mat)


for i in range(col):
    v=HouseHolder_Copy_Row(Mat,i)
    outer=HouseHolder_Row(Mat,v,i)
    print(outer)
    Mat[i:,i:]+=outer
    if(i<=col-2):
        v=HouseHolder_Copy_Col(Mat,i)
        outer=HouseHolder_Col(Mat,v,i)
        Mat[i:,i+1:]+=outer
print(Mat)