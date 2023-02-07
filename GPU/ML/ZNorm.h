//Matrix Mean
#pragma once
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"

void InitializeMatrix(float* matrix, int ny, int nx);
void DisplayMatrix(float* temp, const int ny, const int nx);
void Verification(string name,float* hostC, float* gpuC, const int nx);
void Display(string name, float* temp, const int nx);
void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx);

//CPU Implementations
void cpuMean(float* A, float* mean,const int ny, const int nx);
void cpuStdDev(float* A, float* mean, float* stddev,const int ny, const int nx);
void cpuNormalizeMat(float* A, float* A_norm, float* mean, float* stddev, const int ny, const int nx);

//GPU Functions
//Thread Implementations
__host__ void gpuThreadHelper(float* h_A, float* h_A_Norm,float* h_mean, float* ref_mean,float* h_stddev,float *ref_stddev, float* norm_ref, const int ny, const int nx);
__global__ void ThreadMean(float* g_A, float* g_mean, const int ny, const int nx);
__global__ void ThreadStdDev(float* g_A, float* g_mean, float* g_stddev, const int ny, const int nx);


//GPU Kernel for Transposing the Data Matrix
__global__ void Transpose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx);
__global__ void NormMat_Naive(float* g_A, float* g_A_norm, float* g_mean, float* g_stddev, const int ny, const int nx);
//Using Parallel Reduction Approach
__host__ void gpuPRMeanHelper(float* h_A, float*h_A_norm, float* h_mean, float* ref, float* h_stddev, float* ref_stddev, float* norm_ref, const int ny, const int nx);
__host__ void gpuPRMeanHelper_SingleStream(float* h_A, float* h_A_norm, float* h_mean, float* ref, float* h_stddev, float* ref_stddev, float* norm_ref, const int ny, const int nx);
__host__ void gpuPRMeanHelper_16Stream(float* h_A, float* h_A_Norm, float* h_mean, float* ref, float* h_stddev, float* ref_stddev, float* norm_ref, const int ny, const int nx);
//Kernel for parallel reduction
__global__ void PartialReduction(float* g_Vector, float* g_PartialSum,const int size);
__global__ void PartialReduction_STDDEV(float* g_Vector, float* g_PartialSum, float g_mean, const int size);


__global__ void NormalizeMatrix(float* g_Vector, float* g_Vector_Norm, float mean, float stddev, const int size);
