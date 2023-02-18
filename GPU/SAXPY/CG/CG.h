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


void InitializeMatrix(float *matrix, int ny, int nx);
void Generate_Vector(float* in, int size);
void Diag_Dominant_Opt(float* Mat, int N);
void cpuMatrixVect(float* A_T, float* b, float* b_new, const int ny, const int nx);
void cpuMatrixMult(float* A, float* A_T, float* C, const int ny, const int nx);
void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx);
float Dot_Product(float* in_1,float* in_2, int size);
void Const_Vect_Mult(float* vect, float* out, float scalar, int size);
void cpuVectorAddition(float* A, float* B, float* C, int size);
void vector_subtract(float* in_1, float* in_2, float* out, int size);
void C_G(float* A, float* r, float* r_old, float* d, float* d_old, float* x, float* x_old, float beta, float lamdba, int size, int* iter);
void Verify(float* iter, float* res, int size);
void Display(string name, float* temp, const int nx);


__global__ void MatrixVectorMult(float* g_Matrix, float* g_V, float* g_P, const int Size);
__global__ void d_Const_Vect_Mult(float* vect, float* out, float* scalar, int size);
__global__ void VectAdd(float* g_A, float* g_B, float* g_C, int size);
__global__ void d_Commit_Dot(float* g_Partial_Sum, float* dot, int* flag);
__global__ void d_Dot_Partial(float* in_1, float* in_2, float* hold, float* g_PartialSums, int size);
__global__ void comp_lamba(float* in, float* in_2, float* out,int size, int flag);
__global__ void Copy(float* in, float* out, int size);
__host__ void CG_Helper(float* A, float* ref, float* r, float* r_old, float* d, float* d_old, float* x, float* x_old, float beta, float lamdba, int size,int iter);