#pragma once
#include <iostream>
#include <iomanip>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


#define RANGE_MAX 0.5
#define RANGE_MIN -0.5

void InitializeVector(float* vect, int size);
void VaddVerification(float* hostC, float* gpuC, int size);
void DisplayVector(string name, float* temp, const int size);

void cpuVectorAddition(float* A, float* B, float* C, int size);

__host__ void gpuVaddHelper(float* h_A, float* h_B, float* h_C, float* ref, const int size);
__global__ void VectAdd(float* g_A, float* g_B, float* g_C, int size);

