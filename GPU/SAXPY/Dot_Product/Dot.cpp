#include "Dot.h"


int main(){
    float *v_1, *v_2;
    float dot_CPU{}, dot_GPU{};
    int size=1<<10;
    v_1=new float[size];
    v_2=new float[size];
    Generate_Vector(v_1,size);
    Generate_Vector(v_2,size);
    dot_CPU=Dot_Product(v_1,v_2,size);
    Dot_Helper(v_1,v_2,&dot_CPU,&dot_GPU,size);

    delete[] v_1;
    delete[] v_2;

    return 0;
}