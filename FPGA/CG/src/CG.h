#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <math.h>
#include <cmath>


void InitializeMatrix(int *matrix, int ny, int nx);
void Generate_Vector(int* in, int size);
void Diag_Dominant_Opt(int* Mat, int N);
void cpuMatrixVect(int* A_T, int* b, int* b_new, const int ny, const int nx);
void cpuMatrixMult(int* A, int* A_T, int* C, const int ny, const int nx);
void TransposeOnCPU(int* matrix, int* matrixTranspose, int ny, int nx);
int Dot_Product(int* in_1,int* in_2, int size);
void Const_Vect_Mult(int* vect, int* out, int scalar, int size);
void cpuVectorAddition(int* A, int* B, int* C, int size);
void vector_subtract(int* in_1, int* in_2, int* out, int size);
void C_G(int* A, int* r, int* r_old, int* d, int* d_old, int* x, int* x_old, int beta, int lamdba, int size, int* iter);
void Verify(int* iter, int* res, int size);
void Display(string name, int* temp, const int nx);
int L_2(int* in, int size);
