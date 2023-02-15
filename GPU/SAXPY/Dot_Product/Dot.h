#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cmath>


#define RANGE_MAX 0.5
#define RANGE_MIN -0.5

void Generate_Vector(float* in, int size);
void Verify(float GPU_dot, float CPU_dot);
float Dot_Product(float* in_1,float* in_2, int size);
__host__ void Dot_Helper(float* vect_1, float* vect_2, float* ref, float* dot_GPU, int size);
__global__ void d_Dot_Partial(float* in_1, float* in_2, float* hold, float* g_PartialSums, int size);
__global__ void d_Commit_Dot(float* g_Partial_Sum, float* dot);