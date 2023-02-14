#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define RANGE_MAX 0.5
#define RANGE_MIN -0.5


void Verify(float* GPU_dot, float* CPU_dot);
void Generate_Vector(float* in, int size);
__global__ void d_Commit_L_2(float* g_Partial_Sum, float* mu);
__host__ void L_2_Helper(float* vector, float* ref,float* mu_GPU, int size);
__global__ void d_L_2_Partial_Reduction(float* in, float* hold_vect,float* g_PartialSum, int size);
float L_2(float* in, int size);