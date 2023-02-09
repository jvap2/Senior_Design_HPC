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

void InitializeMatrix(float* matrix, int ny, int nx);
void ZeroMatrix(float* temp, const int ny, const int nx);
void SVDVerification(float* hostC, float* gpuC, const int ny, const int nx);
void DisplayMatrix(string name, float* temp, const int ny, const int nx);
void cpuVectorAddition(float* A, float* B, float* C, int size);
void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx);
void IdentityMatrix(float* temp, int size);
void ZeroVector(float *temp, const int size);
void L_2(float* temp, int k, int size, float& res);
void sign(float x, float& res);
void Copy_To_Row(float*A, float* vect,int k, int nx);
void Copy_Col_House(float* A, float* vect, int k, int ny, int nx);
void Copy_Row_House(float* A, float* vect, int k, int ny, int nx);
void Copy_To_Column(float*A, float* vect, int k, int ny, int nx);
void Dot_Product(float* v_1, float* v_2, float& res, int k, int size);
void Aug_Mat_Vect_Mult(float* A, float* res, float* v, int k, int ny, int nx);
void Aug_Mat_Vect_Mult_Col(float* A, float* res, float* v, float beta, int k, int ny, int nx);
void const_vect_mult(float* v, float constant, int k, int size);
void Outer_Product(float* w, float* v, float* res, int k, int ny, int nx);
void Matrix_Addition(float* A, float* B, int k, int ny, int nx);
void Matrix_Addition_Col(float* A, float* B, int k, int ny, int nx);
void House_Row(float* A, float* p, float* res, float* v, int k, int ny, int nx);
void House_Col(float* A, float* p, float* res, float* v, int k, int ny, int nx);
void House_1(float*A, float* p, float* p_2, float* res_1, float* res_2, float* v_1, float* v_2, int ny, int nx);
void House_2(float*A, float* p, float* p_2, float* res_1, float* res_2, float* v_1, float* v_2, int ny, int nx);

__host__ void Bidiag_Helper_1(float* A, float* ref,  int ny, int nx);
__host__ void Bidiag_Helper_2(float* A, float* ref,  int ny, int nx);
__global__ void final_L_2(float* sum_arr, int size, float* mu);
__global__ void Compute_Beta(float* dot_array, float* beta, int size);
__global__ void Aug_MatrixVectorMult_Row(float* g_Matrix, float* g_V, float* g_P, float* scalar, int k, const int nx, const int ny) ;
__global__ void Aug_MatrixVectorMult_Col(float* g_Matrix, float* g_V, float* g_P, float* scalar,int k, const int nx, const int ny);
__global__ void d_L_2(float* in, float* v, float* hold, int k, int size,int nx);
__global__ void Mat_Add_Col(float* Mat_A, float* Mat_B, float* Mat_C, int ny, int nx, int k);
__global__ void Mat_Add_Row(float* Mat_A, float* Mat_B, float* Mat_C, int ny, int nx, int k);
__global__ void d_Dot_Product(float* v, float* hold, int k, int size);
__global__ void Update_v(float* v, int k, float* mu, int size);
__global__ void d_Outer_Product(float* w, float* v, float* out, int k, int ny, int nx);
__global__ void d_L_2_CH(float* in, float* v, float* hold, int k, int size,int nx);