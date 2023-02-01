#include "Vadd.h"


int main(){
    float *A,*B,*C, *gpuC;
    int size=1<<6;
    cout<<"Vector Size: "<<size<<endl;
    A=new float[size];
    B=new float[size];
    C=new float[size];
    gpuC=new float[size];
    InitializeVector(A,size);
    InitializeVector(B,size);
    cpuVectorAddition(A,B,C,size);
    gpuVaddHelper(A,B,gpuC,C,size);
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] gpuC;
}