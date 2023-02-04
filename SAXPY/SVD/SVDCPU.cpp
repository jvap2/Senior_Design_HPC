#include "SVD.h"

void L_2(float* temp, int k, int size, float& res){
    float sum{};
    for(int i=k; i<size; i++){
        sum+= powf(temp[i], 2.0f);
    }
    res=sqrtf(sum);
    cout<< "L_2: "<<res<<endl;
}

void cpuVectorAddition(float* A, float* B, float* C, int size){
    for(int i=0; i<size; i++){
        C[i]=A[i]+B[i];
        cout<<"C["<<i<<"]="<<C[i]<<endl;
    }
}

void sign(float x, float& res){
    if(x>0){
        res=1;
    }
    else if( x<0 ){
        res=-1;
    }
    else{
        res=0.0f;
    }
}

void Copy_To_Row(float*A, float* vect,int k, int nx){
    /*Here, we will copy v[k+2:nx-1] to A[k][k+2:nx-1], i.e. we are
    overwriting the kth row of A*/
    for (int i=k+2; i<nx; i++){
        A[k*nx+i]=vect[i];
    }
}

void Copy_Col_House(float* A, float* vect, int k, int ny, int nx){
    for (int i=k; i<ny;i++){
        vect[i]=A[i*nx+k];
    }
}

void Copy_To_Column(float*A, float* vect, int k, int ny, int nx){
    /*Here we will be overwriting A[k+1:m-1][j] with u[k+1:m-1]*/
    for (int i=k+1; i<ny;i++){
        A[i*nx+k]=vect[i];
    }
}


void Dot_Product(float* v_1, float* v_2, float& res, int k, int size){
    for(int i=k;i<size; i++){
        res+=v_1[i]*v_2[i];
    }
}

void Aug_Mat_Vect_Mult(float* A, float* res, float* v, int k, int ny, int nx){
    float* p=A;
    float temp{};
    TransposeOnCPU(A,p,ny,nx);
    for(int i=k; i<nx; i++){
        for(int j=k; j<ny; j++){
            temp+=p[i*ny+j]*v[j];
        }
        res[i]=temp;
        temp=0;
    }
}

void const_vect_mult(float* v, float constant, int k, int size){
    for(int i=k; i<size; i++){
        v[i]*=constant;
    }
}

void Outer_Product(float* w, float* v, float* res, int k, int ny, int nx){
    for(int i=k; i<ny; i++){
        for(int j=k; j<nx; j++){
            res[i*nx+j]=v[i]*w[j];
        }
    }
}

void Matrix_Addition(float* A, float* B, float* C, int k, int ny, int nx){
    for(int i=k; i<ny; i++){
        for(int j=k; j<nx; j++){
            C[i*nx+j]=A[i*nx+j]+B[i*nx+j];
        }
    }
}

void House_Row(float* A, float* v, int k, int ny, int nx){
    Copy_Col_House(A,v,k,ny,nx);
    float mu{};
    float beta{};
    float s{};
    float dot{};
    float res[nx-k];
    float* p=A;
    //p will store vw^T
    L_2(v,k,ny,mu);
    sign(v[k],s);
    //The above two lines are for the householder algo
    beta=v[k]+s*mu;
    for(int i=k+1;i<ny;i++){
        v[i]/=beta;
    }
    v[k]=1.0f;//Done with the house, do row.house
    //Row House
    Dot_Product(v,v,dot,k,ny);
    beta=-2/dot;
    Aug_Mat_Vect_Mult(A,res,v,k,ny,nx);
    const_vect_mult(res,beta,k,nx);
    Outer_Product(res,v,p,k,ny,nx);
    Matrix_Addition(A,p,A,k,ny,nx);
    Copy_To_Column(A,v, k,ny,nx);
}




