#include "SVD.h"

float L_2(float* temp, int size, float& res){
    float sum{};
    for(int i{}; i<size; i++){
        sum+= powf(temp[i], 2.0f);
    }
    res=sqrtf(sum);
}

void cpuVectorAddition(float* A, float* B, float* C, int size){
    for(int i=0; i<size; i++){
        C[i]=A[i]+B[i];
    }
}

float sign(float x){
    if(x>0){
        float res=1;
        return res;
    }
    else if( x<0 ){
        float res=-1;
        return res;
    }
}

void Vect_Const_Mult(float* temp, float* return_temp, float a, int size){
    float* return_temp{};
    for(int i=0; i<size; i++){
        return_temp[i]=temp[i]*a;
    }
}

void Vect_Const_Mult_Addr(float* temp, float a, int size){
    for(int i=0; i<size; i++){
        temp[i]*=a;
    }
}

void Update_A_1(float* A, float* u, float* p_hold, float* p_hold_2, int k, int ny, int nx){
    float temp{};
    for(int j=k; j<nx; j++){
        for(int m=k;m<ny;m++){
            temp+=u[m]*A[m*nx+j];
        }
        p_hold[j]=temp;
        temp=0;
    }
    for(int i=k;i<ny;i++){
        for(int j=k; j<nx;j++){
            p_hold_2[i*nx+j]=-2*u[i]*p_hold[j];
        }
    }
    for(int i=k; i<ny;i++){
        for(int j=k;j<nx;j++){
            A[i*nx+j]+=p_hold_2[i*nx+j];
        }
    }
}


void Update_A_2(float* A, float* v, float* p_hold, float* p_hold_2, int k, int ny, int nx){
    float temp{};
    for(int i=k; i<ny;i++){
        for(int j=k+1; j<nx; j++){
            temp+=A[i*nx+j]*v[j];
        }
        p_hold[i]=-2*temp;
        temp=0;
    }
    for(int i=k; i<ny;i++){
        for(int j=k+1; j<nx; j++){
            p_hold_2[i*nx+j]=p_hold[i]*v[j];
        }
    }
    for(int i=k; i<ny;i++){
        for(int j=k;j<nx;j++){
            A[i*nx+j]+=p_hold_2[i*nx+j];
        }
    }


}


void Bidiagonal(float* A, float* x, float* u, float* v, float* e_1, float* e_1_u, float* e_1_v,int ny, int nx){
    float s_x_1{};
    float x_norm{};
    float u_norm{};
    float v_norm{};
    float p_hold[ny];
    float p_hold_2[ny];
    float p_hold_3[ny];
    float p_hold_4[ny];
    for(int k{}; k<nx; k++){
        for(int i=k; i<ny; i++){
            x[i]=A[i*ny+k];
        }
        s_x_1=sign(x[k]);
        L_2(x,ny,x_norm);
        Vect_Const_Mult(e_1,e_1_u,s_x_1*x_norm,ny);
        cpuVectorAddition(x,e_1_u,u,ny);
        L_2(u,ny,u_norm);
        Vect_Const_Mult_Addr(u,1.0f/u_norm,ny);
        Update_A_1(A, u, p_hold, p_hold_2, k, ny, nx);
        if(k<=nx-2){
            for(int i=k+1; i<nx;i++){
                x[i]=A[k*nx+i];
                s_x_1=sign(x[k]);
                L_2(x,ny,x_norm);
                Vect_Const_Mult(e_1,e_1_v,s_x_1*x_norm,ny);
                cpuVectorAddition(x,e_1_u,u,ny);
                L_2(v,ny,v_norm);
                Vect_Const_Mult_Addr(v,1.0f/v_norm,ny);
                Update_A_2(A, v, p_hold_3, p_hold_4, k, ny, nx);
            }
        }
    }
}