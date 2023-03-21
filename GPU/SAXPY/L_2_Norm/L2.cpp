#include "L2.h"


int main(){
    float *v;
    float mu, mu_GPU;
    int size=64;
    cout<<"Vector Size is "<<size<<endl;
    v=new float[size];
    Generate_Vector(v,size);
    chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    mu=L_2(v,size);
    end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
    L_2_Helper(v,&mu,&mu_GPU,size);

    delete[] v;
}