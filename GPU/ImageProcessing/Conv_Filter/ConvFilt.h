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
#include "C/home/sureshm/EE_4830_GPU/Senior_Design_HPC/GPU/ImageProcessing/CImg-3.2.1_pre013023/CImg.h"
using namespace cimg_library;

//GPU Error Handling
#include "GPUErrors.h"
#define FALSE 0

void Filter(CImg<unsigned char>& rgbImg, CImg<unsigned char>& blurImg, int v_off, int h_off);