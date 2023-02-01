#include "Vadd.h"

void InitializeVector(float* vect, int size){
    float *p=vect;
    for (int i=0; i<size; i++){
        p[i] = ((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
    }
}


void VaddVerification(float* hostC, float* gpuC, int size){
    float tol=1e-4;
    float* h=hostC;
    float* g=gpuC;
    for(int i =0; i <size; i++){
        if (fabs(h[i]-g[i])>tol){
            	cout << "Error" << endl;
				cout << "\thostC[" << (i + 1) << "] = " << h[i] << endl;
				cout << "\tgpuC[" << (i + 1) << "] = " << g[i] << endl;
				return;
        }
    }
}

void DisplayVector(string name, float* temp, const int size){
    if (size<6)
	{
		cout << name << endl;
		for (int i = 0; i < size; i++)
		{
				cout << setprecision(6) << temp[i] << "\t";
		}
	}
}