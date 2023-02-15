#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <math.h>

#define RANGE_MAX 0.5
#define RANGE_MIN -0.5

typedef struct{
	float array[4];
}V;


float L_2(float* in, int size);
void Verify(float* GPU_dot, float* CPU_dot);
void Generate_Vector(float* in, int size);
void copy_to_type_V(float* in, V* data, int size);