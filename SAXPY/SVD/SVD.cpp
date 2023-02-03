#include "SVD.h"

int main(){
    float *A, *V, *U, *B;
    int ny=4;
    int nx=3;
    A=new float[ny*nx];
    U= new float[ny*ny];
    V=new float[nx*nx];
    B=new float[ny*nx];
    IdentityMatrix(U,ny);
    IdentityMatrix(V,nx);


    return 0;
}