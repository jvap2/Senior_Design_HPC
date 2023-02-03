#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
#include <math.h>
#include <cmath>


#define RANGE_MAX 0.5
#define RANGE_MIN -0.5

void InitializeMatrix(float* matrix, int ny, int nx);
void ZeroMatrix(float* temp, const int ny, const int nx);
void SVDVerification(float* hostC, float* gpuC, const int ny, const int nx);
void DisplayMatrix(string name, float* temp, const int ny, const int nx);
void cpuVectorAddition(float* A, float* B, float* C, int size);
void L_2(float* temp, int k, int size, float& res);
void sign(float x, float& res);
void eye_1(float* temp, int size);
void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx);
void IdentityMatrix(float* temp, int size);
void ZeroVector(float *temp, const int size);
void Copy_To_Row(float*A, float* vect, int k, int nx);
void Copy_To_Column(float*A, float* vect, int k, int ny, int nx);
void House(float* A, float* temp, int k, int size);

