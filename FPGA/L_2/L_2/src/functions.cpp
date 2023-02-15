#include "/home/sureshm/EE_4830_GPU/Senior_Design_HPC/FPGA/L_2/L_2/src/L_2.hpp"

void Generate_Vector(float* in, int size){
    for(int i{}; i<size;i++){
        in[i]=((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
    }
}

void Verify(float GPU_dot, float CPU_dot){
    float dif=fabsf((GPU_dot)-(CPU_dot));
    if(dif>1e-4){
        cout<<"CPU is "<<(CPU_dot)<<endl;
        cout<<"GPU is "<<(GPU_dot)<<endl;
        cout<<"Error with the Dot Product"<<endl;
        return;
    }
}

void copy_to_type_V(float* in, V* data, int size){
	const int indexes=size/4;
	for(int i{}; i<indexes; i++){
		for(int j{}; j<4; j++){
			data[i].array[j]=in[4*i+j];
		}
	}
}
