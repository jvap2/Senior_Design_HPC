#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define RANGE_MAX 0.5f
#define RANGE_MIN -0.5f

void FE(float* A, float* b, int N);
void Random(float* Mat, int row, int col);
void DisplayMatrix(float* Mat, int N,int Size);
void Diag_Dominant(float* Mat, int N);


__host__ void gpuFEHelper(float* h_Mat, float* h_b, const int N);
__global__ void NaiveFE(float* g_Mat, float* g_b, const int N);
