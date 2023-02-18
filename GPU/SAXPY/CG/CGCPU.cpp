#include "CG.h"

void C_G(float* A, float* r, float* r_old, float* d, float* d_old, float* x, float* x_old, float beta, float lamdba, int size){
    float Ad[size]={};
    float lamd_d[size]={};
    float beta_d[size]={};
    float lambd_AD[size]={};
    float temp_1{};
    float temp_2{};
    int count=0;
    int MaxIter=10*size;
    while(count<MaxIter){
        cpuMatrixVect(A,d_old,Ad,size,size);
        temp_1=Dot_Product(r_old,r_old,size);
        temp_2=Dot_Product(d_old,Ad,size);
        if(fabsf(temp_1)<1e-8 || fabsf(temp_2)<1e-8){
            return;
        }
        lamdba=temp_1/temp_2;
        float neg_lamb=-lamdba;
        Const_Vect_Mult(d_old,lamd_d,lamdba,size);
        Const_Vect_Mult(Ad,lambd_AD,neg_lamb,size);
        cpuVectorAddition(x_old,lamd_d,x,size);
        cpuVectorAddition(r_old,lambd_AD,r,size);
        temp_2=Dot_Product(r,r,size);
        if(fabsf(temp_2)<1e-5){
            return;
        }
        beta=temp_2/temp_1;
        Const_Vect_Mult(d_old,beta_d,beta,size);
        cpuVectorAddition(r,beta_d,d,size);
        for(int i=0;i<size;i++){
            d_old[i]=d[i];
            r_old[i]=r[i];
            x_old[i]=x[i];

        }
        count++;
    }
}