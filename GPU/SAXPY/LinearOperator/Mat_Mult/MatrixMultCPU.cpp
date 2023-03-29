#include "MatrixMult.h"

void cpuMatrixMult(float* A, float* B, float* C, const int ny, const int nx)
{
	float fSum;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	int count=0;
	for (int i = 0; i < ny; i++)
	{
		for (int j = i; j < nx; j++)
		{
			fSum = 0.0f;
			for (int k = 0; k < nx; k++)
			{
				fSum += (A[abs(i-k)] * B[abs(k-j)]);
			}

			C[i*nx+j] = fSum;
			count++;
		}
	}
	//This is better way to collect time, collect it in the function rather than in the main file since it 
	//take time to launch a function
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}

void cpuMarMult_2(float* A, float* B, float* C, const int ny, const int nx, int l, int stop){
	float fSum;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	if(stop>ny){
		stop=ny;
	}
	int count=0;
	for (int i = l; i < stop; i++)
	{
		for (int j = i; j < nx; j++)
		{
			fSum = 0.0f;
			for (int k = 0; k < nx; k++)
			{
				fSum += (A[abs(i-k)] * B[abs(k-j)]);
			}
			C[count] = fSum;
			count++;
		}
	}
	//This is better way to collect time, collect it in the function rather than in the main file since it 
	//take time to launch a function
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}