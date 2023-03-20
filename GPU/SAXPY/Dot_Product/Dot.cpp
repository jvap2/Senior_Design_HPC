#include "Dot.h"


int main(){
    float *v_1, *v_2;
    float dot_CPU{}, dot_GPU{};
    int size=65536;
    v_1=new float[size];
    v_2=new float[size];
    Generate_Vector(v_1,size);
    Generate_Vector(v_2,size);
    chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    dot_CPU=Dot_Product(v_1,v_2,size);
    end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
    Dot_Helper(v_1,v_2,&dot_CPU,&dot_GPU,size);

    delete[] v_1;
    delete[] v_2;

    return 0;
}