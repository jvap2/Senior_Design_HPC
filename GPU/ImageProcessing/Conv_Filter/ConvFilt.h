#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//The clock we used before does not have the necessary resolution for image processing
// We are using a new clock with a higher resolution, this is specified by line 6
//Image Processing Routines
#include "CImg.h"
using namespace cimg_library;

//GPU Error Handling
#include "GPUErrors.h"
#define FALSE 0

void Filter(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w);
__host__ void Helper_Filter(unsigned char* h_in, unsigned char* h_out,unsigned int greySIZE,
	unsigned int blurSIZE,
	unsigned int h,
	unsigned int w);
__global__ void d_Gauss_Filter(unsigned char* in, unsigned char* out,int h, int w);
