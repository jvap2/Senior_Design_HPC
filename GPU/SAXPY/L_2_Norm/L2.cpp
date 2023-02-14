#include "L2.h"


int main(){
    float *v;
    float mu, mu_GPU;
    int size=1<<12;
    cout<<"Vector Size is "<<size<<endl;
    v=new float[size];
    Generate_Vector(v,size);
    mu=L_2(v,size);
    L_2_Helper(v,&mu,&mu_GPU,size);

    delete[] v;
}