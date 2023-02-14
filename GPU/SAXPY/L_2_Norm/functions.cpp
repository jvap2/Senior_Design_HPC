#include "L2.h"


void Generate_Vector(float* in, int size){
    for(int i{}; i<size;i++){
        in[i]=((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
    }
}

void Verify(float GPU_dot, float CPU_dot){
    float dif=fabsf(GPU_dot-CPU_dot);
    if(dif>1e-6){
        cout<<"Error with the Dot Product"<<endl;
        return;
    }
}