#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;

#define RANGE_MAX 0.5f
#define RANGE_MIN -0.5f

void FE(float* A, float* b, int N);
void Random(float* Mat, int row, int col);
void DisplayMatrix(float* Mat, int N,int Size);
void Diag_Dominant(float* Mat, int N);
