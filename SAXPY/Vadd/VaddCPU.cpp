#include "Vadd.h"

void cpuVectorAddition(float* A, float* B, float* C, int size){
    chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    for(int i=0; i<size; i++){
        C[i]=A[i]+B[i];
    }
    end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}