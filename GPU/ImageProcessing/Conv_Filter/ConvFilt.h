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

void Filter(CImg<unsigned char>& rgbImg, CImg<unsigned char>& blurImg, int v_off, int h_off);
void Sharpen_Blue(CImg<unsigned char>& rgbImg, CImg<unsigned char>& blurImg, char* mask, int v_off, int h_off);
void cpu__RGBtoGrayScale_Ver0(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg);
__host__ void Helper_Filter(unsigned char* h_in, unsigned char* h_out,	unsigned int rgbSIZE,
	unsigned int blurSIZE,
	unsigned int h,
	unsigned int w,
    unsigned int h_off,
    unsigned int w_off);
__global__ void d_Filter_v0(unsigned char* in, unsigned char* out,int h, int w, int horiz_off, int vert_off);
__global__ void d_Sharpen(unsigned char* in, unsigned char* out,unsigned char* mask, int h, int w, int horiz_off, int vert_off);
__host__ void Helper_Sharpen(unsigned char* h_in, unsigned char* h_out,	char* mask, unsigned int rgbSIZE,
	unsigned int blurSIZE,
	unsigned int h,
	unsigned int w,
    unsigned int h_off,
    unsigned int w_off);