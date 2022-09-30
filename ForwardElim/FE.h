#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;

#define RANGE_MAX 1.0f
#define RANGE_MIN 0.0f

void FE(float* A, float* b, int N);
void Random(float* Mat, int row, int col);
void DisplayMatrix(float* Mat, int N,int Size);
