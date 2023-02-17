#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


void InitializeMatrix(float *matrix, int ny, int nx);
void Generate_Vector(float* in, int size);
void Diag_Dominant_Opt(float* Mat, int N);
void cpuMatrixVect(float* A_T, float* b, float* b_new, const int ny, const int nx);
void cpuMatrixMult(float* A, float* A_T, float* C, const int ny, const int nx);
void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx);
float Dot_Product(float* in_1,float* in_2, int size);