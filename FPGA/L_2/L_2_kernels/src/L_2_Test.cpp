#include "../Senior_Design_HPC/FPGA/L_2/L_2/src/L_2.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define SIZE 1024
#define part 4
#define part_arr 4*part


int main(){
	const int indexes=SIZE/part;
	float* data=(float*)malloc(SIZE*sizeof(float));
	V *d_data = (V*) malloc(indexes * sizeof(VECTOR));
	float ref=0.0f;
	float d_ref=0.0f;
	Generate_Vector(data, SIZE);
	copy_to_type_V(data,d_data,SIZE);
	ref=L_2(data,SIZE);
	fp_reduce(d_data, &d_ref, SIZE);
	printf("Sum: %0.12f\nReference: %0.12f\n", d_ref, ref);
}
