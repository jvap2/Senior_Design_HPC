#include "SVD.h"

int main(){
    float *v_1, *v_2, *p, *p_2, *res_1, *res_2;
    int ny=4;
    int nx=3;
    p=new float[ny*nx];
    p_2=new float[ny*nx];
    res_1=new float[nx];
    res_2=new float[ny];
    v_1=new float[ny];
    v_2=new float[nx];
    // float A[]={4, 3, 0, 2, 5,2, 1, 2, 1, 6,4, 4, 0, 3, 0,5, 6, 1, 3, 7};
    float A[]={1,2,3,4,5,6,7,8,9,10,11,12};
    DisplayMatrix("Normal A", A, ny, nx);
    if(ny>nx){
        House_1(A,p,p_2,res_1,res_2, v_1, v_2, ny,nx);
    }
    if(ny<nx){
        House_2(A,p,p_2,res_1,res_2, v_1, v_2, ny,nx);
    }
    DisplayMatrix("Bidiag A",A,ny,nx);

    return 0;
}