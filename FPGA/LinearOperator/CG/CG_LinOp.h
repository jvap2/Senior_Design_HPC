#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <math.h>
#include <cmath>
#define N 512


void InitializeMatrix(float *matrix, int ny, int nx);
void Generate_Vector(float* in, int size);
void Diag_Dominant_Opt(float* Mat, int size);
void cpuMatrixVect(float* A_T, float* b, float* b_new, const int ny, const int nx);
void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx);
float Dot_Product(float* in_1,float* in_2, int size);
void Const_Vect_Mult(float* vect, float* out, float scalar, int size);
void cpuVectorAddition(float* A, float* B, float* C, int size);
void vector_subtract(float* in_1, float* in_2, float* out, int size);
void C_G(float* A, float* r, float* r_old, float* d, float* d_old, float* x, float* x_old, float* beta, float* lamdba, int size, int* iter);
void Verify(float* iter, float* res, int size, bool* check, int* count);
void Display(string name, float* temp, const int nx);
float L_2(float* in, int size);

void final(float A[N],float r[N],float r_old[N],float d[N],float d_old[N],float x[N],float x_old[N],float beta[1],float lambda[1],int iter[1]);
void compt(float scal_1, float scal_2, float out[1],int flag);
void const_Vect_mult(float scalar[1], float vect[N], float out[N]);
void Copy(float new_v[N], float old_v[N]);
void MatVec_Mult(float A[N], float b[N], float b_new[N]);
void vadd_p(float a[N], float b[N], float out[N]);
void dot_p(float a[N], float b[N], float out[1]);
