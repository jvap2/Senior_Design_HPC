#include "CG.h"
#include "GPUErrors.h"


int main(){
    float *A,*A_T,*A_res, *b, *b_res, *r, *r_old, *d, *d_old, *x, *x_old, *Ax;
    float lambda{}, beta{};
    int ny=5;
    int nx=5;
    
    Ax=new float[ny];
    A=new float[ny*nx];
    A_T=new float[ny*nx];
    A_res=new float[ny*nx];
    b=new float[ny];
    b_res=new float[ny];
    r=new float[ny];
    r_old=new float[ny];
    d=new float[ny];
    d_old=new float[ny];
    x=new float[ny];
    x_old=new float[ny];
    InitializeMatrix(A,ny,nx);
    TransposeOnCPU(A,A_T,ny,ny);
    cpuMatrixMult(A,A_T,A_res,ny,nx);
    cpuMatrixVect(A_T,b,b_res,ny,nx);
    Generate_Vector(x_old,ny);
    cpuMatrixVect(A_res, x_old, Ax, ny, nx);
    vector_subtract(b_res,Ax,r_old,ny);
    C_G(A_res,r,r_old,d,d_old,x,x_old,beta,lambda,ny);

    delete[] Ax;
    delete[] A;
    delete[] A_T;
    delete[] A_res;
    delete[] b;
    delete[] b_res;
    delete[] r;
    delete[] r_old;
    delete[] d;
    delete[] d_old;
    delete[] x;
    delete[] x_old;

    return 0;

}