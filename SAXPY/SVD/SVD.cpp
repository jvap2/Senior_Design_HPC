#include "SVD.h"

int main(){
    float *x,*u,*v,*e_1,*e_1_u,*e_1_v;
    // float* A;
    int rows=4;
    int cols=5;
    // A=new float[rows*cols];
    float A[]={4, 3, 0, 2, 5, 2, 1, 2, 1, 6, 4, 4, 0, 3, 0, 5, 6, 1, 3, 7};
    x=new float[cols];
    u=new float[cols];
    v=new float[rows];
    e_1=new float[cols];
    e_1_u=new float[cols];
    e_1_v=new float[rows];
    ZeroVector(x,cols);
    ZeroVector(u,cols);
    ZeroVector(v,rows);
    ZeroVector(e_1,cols);
    ZeroVector(e_1_u,cols);
    ZeroVector(e_1_v,rows);
    e_1[0]=1;
    Bidiagonal(A, x, u, v, e_1, e_1_u,e_1_v,rows, cols);
    for(int i{}; i<rows; i++){
        for(int j{}; j<cols; j++){
            cout<<A[i*cols+j]<<'\t';
        }
        cout<<endl;
    }
}