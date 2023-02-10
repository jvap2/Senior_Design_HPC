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
//CPU Functions
float cpu__RGBtoGrayScale_Ver0(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg);
//First argument is the image we have to pass, we are using reference as well so we are not copying everytime, hence why we pass
//by reference. Second image has the output image. This is the naive version
float cpu__RGBtoGrayScale_Ver1(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w);
//Now, we are passing pointers to access the input and output faster
//GPU Helper Function
__host__ double gpu_RGBtoGrayScaleHelper(unsigned char* h_in, unsigned char* h_out, unsigned int rgbSIZE,
	unsigned int graySIZE,
	unsigned int h, 
	unsigned int w,
	unsigned int kernelVer);

//GPU Kernels
__global__ void gpu_RGBtoGrayScaleVer0(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w);
__global__ void gpu_RGBtoGrayScaleVer1(unsigned char* r, 
	unsigned char* g,
	unsigned char* b,
	unsigned char* out, unsigned int h, unsigned int w);
__global__ void gpu_RGBtoGrayScaleVer2(unsigned char* r,
	unsigned char* g,
	unsigned char* b,
	unsigned char* out, unsigned int h, unsigned int w);





