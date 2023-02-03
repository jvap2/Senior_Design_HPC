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
float L_2(float* temp, int size, float& res);
float sign(float x);
void ZeroVector(float *temp, const int size);
void Vect_Const_Mult_Addr(float* temp, float a, int size);
void Vect_Const_Mult(float* temp, float* return_temp, float a, int size);
void Bidiagonal(float* A, float* x, float* u, float* v, float* e_1, float* e_1_u, float* e_1_v, int ny, int nx);