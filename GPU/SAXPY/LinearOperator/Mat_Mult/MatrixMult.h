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

void InitializeMatrix(float* matrix, int ny, int nx);
void ZeroMatrix(float* temp, const int ny, const int nx);
void MatrixMultVerification(float* hostC, float* gpuC, const int ny, const int nx);
void DisplayMatrix(string name, float* temp, const int ny, const int nx);

//CPU Implementations
void cpuMatrixMult(float* A, float* B, float* C, const int ny, const int nx);
void Generate_Vector(float* in, int size);
void Diag_Dominant_Opt(float* Mat, int N);
void Verify(float* iter, float* res, int size);



//GPU Functions
__host__ void gpuMultHelper(float* h_A, float* h_B, float* h_C, float* ref, const int ny, const int nx);
__global__ void NaiveMult(float* g_A, float* g_B, float* g_C, const int ny, const int nx);
//Note, we have named this NaiveMult, this is not optimized
