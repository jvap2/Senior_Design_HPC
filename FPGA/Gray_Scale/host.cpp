/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include "CImg.h"
using namespace cimg_library;
//#define DATA_SIZE 4096

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
//    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    cl_int err;
    cl::Context context;
    cl::Context context2;
    cl::Context context3;
    cl::Context context4;
    cl::Context context5;
    cl::Context context6;
    cl::Context context7;
    cl::Context context8;
    cl::Context context9;
    cl::Context context10;
    cl::Context context11;
    cl::Context context12;
    cl::Context context13;
    cl::Context context14;
    cl::Context context15;
    cl::Context context16;
    cl::Context context17;
    cl::Context context18;
    cl::Context context19;
    cl::Context context20;
    cl::Context context21;
    cl::Context context22;
    cl::Context context23;
    cl::Context context24;
    cl::Context context25;
    cl::Context context26;
    cl::Context context27;
    cl::Context context28;
    cl::Context context29;
    cl::Context context30;
    cl::Context context31;
    cl::Context context32;

    cl::Kernel krnl_vector_add;
    cl::Kernel krnl_vector_add2;
    cl::Kernel krnl_vector_add3;
    cl::Kernel krnl_vector_add4;
    cl::Kernel krnl_vector_add5;
    cl::Kernel krnl_vector_add6;
    cl::Kernel krnl_vector_add7;
    cl::Kernel krnl_vector_add8;
    cl::Kernel krnl_vector_add9;
    cl::Kernel krnl_vector_add10;
    cl::Kernel krnl_vector_add11;
    cl::Kernel krnl_vector_add12;
    cl::Kernel krnl_vector_add13;
    cl::Kernel krnl_vector_add14;
    cl::Kernel krnl_vector_add15;
    cl::Kernel krnl_vector_add16;
    cl::Kernel krnl_vector_add17;
    cl::Kernel krnl_vector_add18;
    cl::Kernel krnl_vector_add19;
    cl::Kernel krnl_vector_add20;
    cl::Kernel krnl_vector_add21;
    cl::Kernel krnl_vector_add22;
    cl::Kernel krnl_vector_add23;
    cl::Kernel krnl_vector_add24;
    cl::Kernel krnl_vector_add25;
    cl::Kernel krnl_vector_add26;
    cl::Kernel krnl_vector_add27;
    cl::Kernel krnl_vector_add28;
    cl::Kernel krnl_vector_add29;
    cl::Kernel krnl_vector_add30;
    cl::Kernel krnl_vector_add31;
    cl::Kernel krnl_vector_add32;


    cl::CommandQueue q;
    cl::CommandQueue q2;
    cl::CommandQueue q3;
    cl::CommandQueue q4;
    cl::CommandQueue q5;
    cl::CommandQueue q6;
    cl::CommandQueue q7;
    cl::CommandQueue q8;
    cl::CommandQueue q9;
    cl::CommandQueue q10;
    cl::CommandQueue q11;
    cl::CommandQueue q12;
    cl::CommandQueue q13;
    cl::CommandQueue q14;
    cl::CommandQueue q15;
    cl::CommandQueue q16;
    cl::CommandQueue q17;
    cl::CommandQueue q18;
    cl::CommandQueue q19;
    cl::CommandQueue q20;
    cl::CommandQueue q21;
    cl::CommandQueue q22;
    cl::CommandQueue q23;
    cl::CommandQueue q24;
    cl::CommandQueue q25;
    cl::CommandQueue q26;
    cl::CommandQueue q27;
    cl::CommandQueue q28;
    cl::CommandQueue q29;
    cl::CommandQueue q30;
    cl::CommandQueue q31;
    cl::CommandQueue q32;

    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
    // hood user ptr
    // is used if it is properly aligned. when not aligned, runtime had no choice
    // but to create
    // its own host side buffer. So it is recommended to use this allocator if
    // user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
    // boundary. It will
    // ensure that user buffer is used when user create Buffer/Mem object with
    // CL_MEM_USE_HOST_PTR

	CImg<unsigned char> imgRGB = CImg<>("/home/sureshm/Senior_Design_HPC/GPU/ImageProcessing/Gray_Scale/mountain-landscape-reflection.jpg");
	unsigned char* ptrRGB = imgRGB.data();
	CImg<unsigned char>imgGrayScale(imgRGB.width(),imgRGB.height());
	unsigned char* ptrGray = imgGrayScale.data();
	size_t rgbSize = imgRGB.width() * imgRGB.height() * 3*sizeof(unsigned char);
	size_t graySize = imgRGB.width() * imgRGB.height() * 1 * sizeof(unsigned char);
	unsigned int stride =imgRGB.width() * imgRGB.height();
	int gsize=((stride/32)+1);
	unsigned int gsize_b=((stride/32)+1)*sizeof(unsigned char);
	unsigned char ptrg1[gsize]{};
	unsigned char ptrg2[gsize]{};
	unsigned char ptrg3[gsize]{};
	unsigned char ptrg4[gsize]{};
	unsigned char ptrg5[gsize]{};
	unsigned char ptrg6[gsize]{};
	unsigned char ptrg7[gsize]{};
	unsigned char ptrg8[gsize]{};
	unsigned char ptrg9[gsize]{};
	unsigned char ptrg10[gsize]{};
	unsigned char ptrg11[gsize]{};
	unsigned char ptrg12[gsize]{};
	unsigned char ptrg13[gsize]{};
	unsigned char ptrg14[gsize]{};
	unsigned char ptrg15[gsize]{};
	unsigned char ptrg16[gsize]{};
	unsigned char ptrg17[gsize]{};
	unsigned char ptrg18[gsize]{};
	unsigned char ptrg19[gsize]{};
	unsigned char ptrg20[gsize]{};
	unsigned char ptrg21[gsize]{};
	unsigned char ptrg22[gsize]{};
	unsigned char ptrg23[gsize]{};
	unsigned char ptrg24[gsize]{};
	unsigned char ptrg25[gsize]{};
	unsigned char ptrg26[gsize]{};
	unsigned char ptrg27[gsize]{};
	unsigned char ptrg28[gsize]{};
	unsigned char ptrg29[gsize]{};
	unsigned char ptrg30[gsize]{};
	unsigned char ptrg31[gsize]{};
	unsigned char ptrg32[gsize]{};
	for (int j=0; j<gsize; j++){
		ptrg1[j]=ptrGray[j];
		ptrg2[j]=ptrGray[j+ gsize];
		ptrg3[j]=ptrGray[j+ 2*gsize];
		ptrg4[j]=ptrGray[j+3*gsize];
		ptrg5[j]=ptrGray[j+4*gsize];
		ptrg6[j]=ptrGray[j+5*gsize];
		ptrg7[j]=ptrGray[j+6*gsize];
		ptrg8[j]=ptrGray[j+7*gsize];
		ptrg9[j]=ptrGray[j+8*gsize];
		ptrg10[j]=ptrGray[j+ 9*gsize];
		ptrg11[j]=ptrGray[j+ 10*gsize];
		ptrg12[j]=ptrGray[j+11*gsize];
		ptrg13[j]=ptrGray[j+12*gsize];
		ptrg14[j]=ptrGray[j+13*gsize];
		ptrg15[j]=ptrGray[j+14*gsize];
		ptrg16[j]=ptrGray[j+15*gsize];
		ptrg17[j]=ptrGray[j+16*gsize];
		ptrg18[j]=ptrGray[j+ 17*gsize];
		ptrg19[j]=ptrGray[j+ 18*gsize];
		ptrg20[j]=ptrGray[j+19*gsize];
		ptrg21[j]=ptrGray[j+20*gsize];
		ptrg22[j]=ptrGray[j+21*gsize];
		ptrg23[j]=ptrGray[j+22*gsize];
		ptrg24[j]=ptrGray[j+23*gsize];
		ptrg25[j]=ptrGray[j+24*gsize];
		ptrg26[j]=ptrGray[j+ 25*gsize];
		ptrg27[j]=ptrGray[j+ 26*gsize];
		ptrg28[j]=ptrGray[j+27*gsize];
		ptrg29[j]=ptrGray[j+28*gsize];
		ptrg30[j]=ptrGray[j+29*gsize];
		ptrg31[j]=ptrGray[j+30*gsize];
		if(j+31*gsize<imgRGB.width() * imgRGB.height()){
			ptrg32[j]=ptrGray[j+31*gsize];
		}
	}
    // Create the test data

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context2 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context3 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context4 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context5 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context6 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context7 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context8 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context9 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context10 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context11 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context12 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context13 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context14 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context15 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context16 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context17 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context18 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context19 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context20 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context21 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context22 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context23 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context24 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context25 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context26 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context27 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context28 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context29 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context30 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context31 = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, context32 = cl::Context(device, nullptr, nullptr, nullptr, &err));

        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q2 = cl::CommandQueue(context2, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q3 = cl::CommandQueue(context3, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q4 = cl::CommandQueue(context4, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q5 = cl::CommandQueue(context5, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q6 = cl::CommandQueue(context6, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q7 = cl::CommandQueue(context7, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q8 = cl::CommandQueue(context8, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q9 = cl::CommandQueue(context9, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q10 = cl::CommandQueue(context10, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q11 = cl::CommandQueue(context11, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q12 = cl::CommandQueue(context12, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q13 = cl::CommandQueue(context13, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q14 = cl::CommandQueue(context14, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q15 = cl::CommandQueue(context15, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q16 = cl::CommandQueue(context16, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q17 = cl::CommandQueue(context17, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q18 = cl::CommandQueue(context18, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q19 = cl::CommandQueue(context19, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q20 = cl::CommandQueue(context20, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q21 = cl::CommandQueue(context21, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q22 = cl::CommandQueue(context22, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q23 = cl::CommandQueue(context23, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q24 = cl::CommandQueue(context24, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q25 = cl::CommandQueue(context25, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q26 = cl::CommandQueue(context26, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q27 = cl::CommandQueue(context27, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q28 = cl::CommandQueue(context28, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q29 = cl::CommandQueue(context29, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q30 = cl::CommandQueue(context30, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q31 = cl::CommandQueue(context31, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q32 = cl::CommandQueue(context32, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        cl::Program program2(context2, {device}, bins, nullptr, &err);
        cl::Program program3(context3, {device}, bins, nullptr, &err);
        cl::Program program4(context4, {device}, bins, nullptr, &err);
        cl::Program program5(context5, {device}, bins, nullptr, &err);
        cl::Program program6(context6, {device}, bins, nullptr, &err);
        cl::Program program7(context7, {device}, bins, nullptr, &err);
        cl::Program program8(context8, {device}, bins, nullptr, &err);
        cl::Program program9(context9, {device}, bins, nullptr, &err);
        cl::Program program10(context10, {device}, bins, nullptr, &err);
        cl::Program program11(context11, {device}, bins, nullptr, &err);
        cl::Program program12(context12, {device}, bins, nullptr, &err);
        cl::Program program13(context13, {device}, bins, nullptr, &err);
        cl::Program program14(context14, {device}, bins, nullptr, &err);
        cl::Program program15(context15, {device}, bins, nullptr, &err);
        cl::Program program16(context16, {device}, bins, nullptr, &err);
        cl::Program program17(context17, {device}, bins, nullptr, &err);
        cl::Program program18(context18, {device}, bins, nullptr, &err);
        cl::Program program19(context19, {device}, bins, nullptr, &err);
        cl::Program program20(context20, {device}, bins, nullptr, &err);
        cl::Program program21(context21, {device}, bins, nullptr, &err);
        cl::Program program22(context22, {device}, bins, nullptr, &err);
        cl::Program program23(context23, {device}, bins, nullptr, &err);
        cl::Program program24(context24, {device}, bins, nullptr, &err);
        cl::Program program25(context25, {device}, bins, nullptr, &err);
        cl::Program program26(context26, {device}, bins, nullptr, &err);
        cl::Program program27(context27, {device}, bins, nullptr, &err);
        cl::Program program28(context28, {device}, bins, nullptr, &err);
        cl::Program program29(context29, {device}, bins, nullptr, &err);
        cl::Program program30(context30, {device}, bins, nullptr, &err);
        cl::Program program31(context31, {device}, bins, nullptr, &err);
        cl::Program program32(context32, {device}, bins, nullptr, &err);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_1}", &err));
            OCL_CHECK(err, krnl_vector_add2 = cl::Kernel(program2, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_2}", &err));
            OCL_CHECK(err, krnl_vector_add3 = cl::Kernel(program3, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_3}", &err));
            OCL_CHECK(err, krnl_vector_add4 = cl::Kernel(program4, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_4}", &err));
            OCL_CHECK(err, krnl_vector_add5 = cl::Kernel(program5, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_5}", &err));
            OCL_CHECK(err, krnl_vector_add6 = cl::Kernel(program6, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_6}", &err));
            OCL_CHECK(err, krnl_vector_add7 = cl::Kernel(program7, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_7}", &err));
            OCL_CHECK(err, krnl_vector_add8 = cl::Kernel(program8, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_8}", &err));
            OCL_CHECK(err, krnl_vector_add9 = cl::Kernel(program9, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_9}", &err));
            OCL_CHECK(err, krnl_vector_add10 = cl::Kernel(program10, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_10}", &err));
            OCL_CHECK(err, krnl_vector_add11 = cl::Kernel(program11, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_11}", &err));
            OCL_CHECK(err, krnl_vector_add12 = cl::Kernel(program12, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_12}", &err));
            OCL_CHECK(err, krnl_vector_add13 = cl::Kernel(program13, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_13}", &err));
            OCL_CHECK(err, krnl_vector_add14 = cl::Kernel(program14, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_14}", &err));
            OCL_CHECK(err, krnl_vector_add15 = cl::Kernel(program15, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_15}", &err));
            OCL_CHECK(err, krnl_vector_add16 = cl::Kernel(program16, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_16}", &err));
            OCL_CHECK(err, krnl_vector_add17 = cl::Kernel(program17, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_17}", &err));
            OCL_CHECK(err, krnl_vector_add18 = cl::Kernel(program18, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_18}", &err));
            OCL_CHECK(err, krnl_vector_add19 = cl::Kernel(program19, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_19}", &err));
            OCL_CHECK(err, krnl_vector_add20 = cl::Kernel(program20, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_20}", &err));
            OCL_CHECK(err, krnl_vector_add21 = cl::Kernel(program21, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_21}", &err));
            OCL_CHECK(err, krnl_vector_add22 = cl::Kernel(program22, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_22}", &err));
            OCL_CHECK(err, krnl_vector_add23 = cl::Kernel(program23, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_23}", &err));
            OCL_CHECK(err, krnl_vector_add24 = cl::Kernel(program24, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_24}", &err));
            OCL_CHECK(err, krnl_vector_add25 = cl::Kernel(program25, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_25}", &err));
            OCL_CHECK(err, krnl_vector_add26 = cl::Kernel(program26, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_26}", &err));
            OCL_CHECK(err, krnl_vector_add27 = cl::Kernel(program27, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_27}", &err));
            OCL_CHECK(err, krnl_vector_add28 = cl::Kernel(program28, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_28}", &err));
            OCL_CHECK(err, krnl_vector_add29 = cl::Kernel(program29, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_29}", &err));
            OCL_CHECK(err, krnl_vector_add30 = cl::Kernel(program30, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_30}", &err));
            OCL_CHECK(err, krnl_vector_add31 = cl::Kernel(program31, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_31}", &err));
            OCL_CHECK(err, krnl_vector_add32 = cl::Kernel(program32, "RGBtoGrayScale_Ver1:{RGBtoGrayScale_Ver1_32}", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication

    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
    									ptrg1, &err));
    OCL_CHECK(err, cl::Buffer buffer_in3(context2, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in4(context2, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg2, &err));
    OCL_CHECK(err, cl::Buffer buffer_in5(context3, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in6(context3, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg3, &err));
    OCL_CHECK(err, cl::Buffer buffer_in7(context4, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in8(context4, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg4, &err));
    OCL_CHECK(err, cl::Buffer buffer_in9(context5, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in10(context5, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg5, &err));
    OCL_CHECK(err, cl::Buffer buffer_in11(context6, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in12(context6, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg6, &err));
    OCL_CHECK(err, cl::Buffer buffer_in13(context7, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in14(context7, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg7, &err));
    OCL_CHECK(err, cl::Buffer buffer_in15(context8, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in16(context8, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg8, &err));
    OCL_CHECK(err, cl::Buffer buffer_in17(context9, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in18(context9, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
    									ptrg9, &err));
    OCL_CHECK(err, cl::Buffer buffer_in19(context10, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in20(context10, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg10, &err));
    OCL_CHECK(err, cl::Buffer buffer_in21(context11, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in22(context11, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg11, &err));
    OCL_CHECK(err, cl::Buffer buffer_in23(context12, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in24(context12, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg12, &err));
    OCL_CHECK(err, cl::Buffer buffer_in25(context13, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in26(context13, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg13, &err));
    OCL_CHECK(err, cl::Buffer buffer_in27(context14, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in28(context14, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg14, &err));
    OCL_CHECK(err, cl::Buffer buffer_in29(context15, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in30(context15, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg15, &err));
    OCL_CHECK(err, cl::Buffer buffer_in31(context16, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in32(context16, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg16, &err));
    OCL_CHECK(err, cl::Buffer buffer_in33(context17, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in34(context17, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
    									ptrg17, &err));
    OCL_CHECK(err, cl::Buffer buffer_in35(context18, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in36(context18, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg18, &err));
    OCL_CHECK(err, cl::Buffer buffer_in37(context19, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in38(context19, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg19, &err));
    OCL_CHECK(err, cl::Buffer buffer_in39(context20, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in40(context20, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg20, &err));
    OCL_CHECK(err, cl::Buffer buffer_in41(context21, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in42(context21, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg21, &err));
    OCL_CHECK(err, cl::Buffer buffer_in43(context22, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in44(context22, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg22, &err));
    OCL_CHECK(err, cl::Buffer buffer_in45(context23, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in46(context23, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg23, &err));
    OCL_CHECK(err, cl::Buffer buffer_in47(context24, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in48(context24, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg24, &err));
    OCL_CHECK(err, cl::Buffer buffer_in49(context25, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in50(context25, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
    									ptrg25, &err));
    OCL_CHECK(err, cl::Buffer buffer_in51(context26, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in52(context26, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg26, &err));
    OCL_CHECK(err, cl::Buffer buffer_in53(context27, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in54(context27, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg27, &err));
    OCL_CHECK(err, cl::Buffer buffer_in55(context28, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in56(context28, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg28, &err));
    OCL_CHECK(err, cl::Buffer buffer_in57(context29, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in58(context29, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg29, &err));
    OCL_CHECK(err, cl::Buffer buffer_in59(context30, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in60(context30, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg30, &err));
    OCL_CHECK(err, cl::Buffer buffer_in61(context31, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in62(context31, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg31, &err));
    OCL_CHECK(err, cl::Buffer buffer_in63(context31, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, rgbSize,
                                         ptrRGB, &err));
    OCL_CHECK(err, cl::Buffer buffer_in64(context8, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, gsize_b,
                                         ptrg32, &err));




    OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_in1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, buffer_in2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add.setArg(4, 0));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(0, buffer_in3));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(1, buffer_in4));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add2.setArg(4, 1));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(0, buffer_in5));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(1, buffer_in6));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add3.setArg(4, 2));
    OCL_CHECK(err, err = krnl_vector_add4.setArg(0, buffer_in7));
    OCL_CHECK(err, err = krnl_vector_add4.setArg(1, buffer_in8));
    OCL_CHECK(err, err = krnl_vector_add4.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add4.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add4.setArg(4, 3));
    OCL_CHECK(err, err = krnl_vector_add5.setArg(0, buffer_in9));
    OCL_CHECK(err, err = krnl_vector_add5.setArg(1, buffer_in10));
    OCL_CHECK(err, err = krnl_vector_add5.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add5.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add5.setArg(4, 4));
    OCL_CHECK(err, err = krnl_vector_add6.setArg(0, buffer_in11));
    OCL_CHECK(err, err = krnl_vector_add6.setArg(1, buffer_in12));
    OCL_CHECK(err, err = krnl_vector_add6.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add6.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add6.setArg(4, 5));
    OCL_CHECK(err, err = krnl_vector_add7.setArg(0, buffer_in13));
    OCL_CHECK(err, err = krnl_vector_add7.setArg(1, buffer_in14));
    OCL_CHECK(err, err = krnl_vector_add7.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add7.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add7.setArg(4, 6));
    OCL_CHECK(err, err = krnl_vector_add8.setArg(0, buffer_in15));
    OCL_CHECK(err, err = krnl_vector_add8.setArg(1, buffer_in16));
    OCL_CHECK(err, err = krnl_vector_add8.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add8.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add8.setArg(4, 7));
    OCL_CHECK(err, err = krnl_vector_add9.setArg(0, buffer_in17));
    OCL_CHECK(err, err = krnl_vector_add9.setArg(1, buffer_in18));
    OCL_CHECK(err, err = krnl_vector_add9.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add9.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add9.setArg(4, 8));
    OCL_CHECK(err, err = krnl_vector_add10.setArg(0, buffer_in19));
    OCL_CHECK(err, err = krnl_vector_add10.setArg(1, buffer_in20));
    OCL_CHECK(err, err = krnl_vector_add10.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add10.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add10.setArg(4, 9));
    OCL_CHECK(err, err = krnl_vector_add11.setArg(0, buffer_in21));
    OCL_CHECK(err, err = krnl_vector_add11.setArg(1, buffer_in22));
    OCL_CHECK(err, err = krnl_vector_add11.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add11.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add11.setArg(4, 10));
    OCL_CHECK(err, err = krnl_vector_add12.setArg(0, buffer_in23));
    OCL_CHECK(err, err = krnl_vector_add12.setArg(1, buffer_in24));
    OCL_CHECK(err, err = krnl_vector_add12.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add12.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add12.setArg(4, 11));
    OCL_CHECK(err, err = krnl_vector_add13.setArg(0, buffer_in25));
    OCL_CHECK(err, err = krnl_vector_add13.setArg(1, buffer_in26));
    OCL_CHECK(err, err = krnl_vector_add13.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add13.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add13.setArg(4, 12));
    OCL_CHECK(err, err = krnl_vector_add14.setArg(0, buffer_in27));
    OCL_CHECK(err, err = krnl_vector_add14.setArg(1, buffer_in28));
    OCL_CHECK(err, err = krnl_vector_add14.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add14.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add14.setArg(4, 13));
    OCL_CHECK(err, err = krnl_vector_add15.setArg(0, buffer_in29));
    OCL_CHECK(err, err = krnl_vector_add15.setArg(1, buffer_in30));
    OCL_CHECK(err, err = krnl_vector_add15.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add15.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add15.setArg(4, 14));
    OCL_CHECK(err, err = krnl_vector_add16.setArg(0, buffer_in31));
    OCL_CHECK(err, err = krnl_vector_add16.setArg(1, buffer_in32));
    OCL_CHECK(err, err = krnl_vector_add16.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add16.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add16.setArg(4, 15));
    OCL_CHECK(err, err = krnl_vector_add17.setArg(0, buffer_in33));
    OCL_CHECK(err, err = krnl_vector_add17.setArg(1, buffer_in34));
    OCL_CHECK(err, err = krnl_vector_add17.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add17.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add17.setArg(4, 16));
    OCL_CHECK(err, err = krnl_vector_add18.setArg(0, buffer_in35));
    OCL_CHECK(err, err = krnl_vector_add18.setArg(1, buffer_in36));
    OCL_CHECK(err, err = krnl_vector_add18.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add18.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add18.setArg(4, 17));
    OCL_CHECK(err, err = krnl_vector_add19.setArg(0, buffer_in37));
    OCL_CHECK(err, err = krnl_vector_add19.setArg(1, buffer_in38));
    OCL_CHECK(err, err = krnl_vector_add19.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add19.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add19.setArg(4, 18));
    OCL_CHECK(err, err = krnl_vector_add20.setArg(0, buffer_in39));
    OCL_CHECK(err, err = krnl_vector_add20.setArg(1, buffer_in40));
    OCL_CHECK(err, err = krnl_vector_add20.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add20.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add20.setArg(4, 19));
    OCL_CHECK(err, err = krnl_vector_add21.setArg(0, buffer_in41));
    OCL_CHECK(err, err = krnl_vector_add21.setArg(1, buffer_in42));
    OCL_CHECK(err, err = krnl_vector_add21.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add21.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add21.setArg(4, 20));
    OCL_CHECK(err, err = krnl_vector_add22.setArg(0, buffer_in43));
    OCL_CHECK(err, err = krnl_vector_add22.setArg(1, buffer_in44));
    OCL_CHECK(err, err = krnl_vector_add22.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add22.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add22.setArg(4, 21));
    OCL_CHECK(err, err = krnl_vector_add23.setArg(0, buffer_in45));
    OCL_CHECK(err, err = krnl_vector_add23.setArg(1, buffer_in46));
    OCL_CHECK(err, err = krnl_vector_add23.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add23.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add23.setArg(4, 22));
    OCL_CHECK(err, err = krnl_vector_add24.setArg(0, buffer_in47));
    OCL_CHECK(err, err = krnl_vector_add24.setArg(1, buffer_in48));
    OCL_CHECK(err, err = krnl_vector_add24.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add24.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add24.setArg(4, 23));
    OCL_CHECK(err, err = krnl_vector_add25.setArg(0, buffer_in49));
    OCL_CHECK(err, err = krnl_vector_add25.setArg(1, buffer_in50));
    OCL_CHECK(err, err = krnl_vector_add25.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add25.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add25.setArg(4, 24));
    OCL_CHECK(err, err = krnl_vector_add26.setArg(0, buffer_in51));
    OCL_CHECK(err, err = krnl_vector_add26.setArg(1, buffer_in52));
    OCL_CHECK(err, err = krnl_vector_add26.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add26.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add26.setArg(4, 25));
    OCL_CHECK(err, err = krnl_vector_add27.setArg(0, buffer_in53));
    OCL_CHECK(err, err = krnl_vector_add27.setArg(1, buffer_in54));
    OCL_CHECK(err, err = krnl_vector_add27.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add27.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add27.setArg(4, 26));
    OCL_CHECK(err, err = krnl_vector_add28.setArg(0, buffer_in55));
    OCL_CHECK(err, err = krnl_vector_add28.setArg(1, buffer_in56));
    OCL_CHECK(err, err = krnl_vector_add28.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add28.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add28.setArg(4, 27));
    OCL_CHECK(err, err = krnl_vector_add29.setArg(0, buffer_in57));
    OCL_CHECK(err, err = krnl_vector_add29.setArg(1, buffer_in58));
    OCL_CHECK(err, err = krnl_vector_add29.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add29.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add29.setArg(4, 28));
    OCL_CHECK(err, err = krnl_vector_add30.setArg(0, buffer_in59));
    OCL_CHECK(err, err = krnl_vector_add30.setArg(1, buffer_in60));
    OCL_CHECK(err, err = krnl_vector_add30.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add30.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add30.setArg(4, 29));
    OCL_CHECK(err, err = krnl_vector_add31.setArg(0, buffer_in61));
    OCL_CHECK(err, err = krnl_vector_add31.setArg(1, buffer_in62));
    OCL_CHECK(err, err = krnl_vector_add31.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add31.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add31.setArg(4, 30));
    OCL_CHECK(err, err = krnl_vector_add32.setArg(0, buffer_in63));
    OCL_CHECK(err, err = krnl_vector_add32.setArg(1, buffer_in64));
    OCL_CHECK(err, err = krnl_vector_add32.setArg(2, imgRGB.height()));
    OCL_CHECK(err, err = krnl_vector_add32.setArg(3, imgRGB.width()));
    OCL_CHECK(err, err = krnl_vector_add32.setArg(4, 31));



    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q2.enqueueMigrateMemObjects({buffer_in3, buffer_in4}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q3.enqueueMigrateMemObjects({buffer_in5, buffer_in6}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q4.enqueueMigrateMemObjects({buffer_in7, buffer_in8}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q5.enqueueMigrateMemObjects({buffer_in9, buffer_in10}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q6.enqueueMigrateMemObjects({buffer_in11, buffer_in12}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q7.enqueueMigrateMemObjects({buffer_in13, buffer_in14}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q8.enqueueMigrateMemObjects({buffer_in15, buffer_in16}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q9.enqueueMigrateMemObjects({buffer_in17, buffer_in18}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q10.enqueueMigrateMemObjects({buffer_in19, buffer_in20}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q11.enqueueMigrateMemObjects({buffer_in21, buffer_in22}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q12.enqueueMigrateMemObjects({buffer_in23, buffer_in24}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q13.enqueueMigrateMemObjects({buffer_in25, buffer_in26}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q14.enqueueMigrateMemObjects({buffer_in27, buffer_in28}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q15.enqueueMigrateMemObjects({buffer_in29, buffer_in30}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q16.enqueueMigrateMemObjects({buffer_in31, buffer_in32}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q17.enqueueMigrateMemObjects({buffer_in33, buffer_in34}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q18.enqueueMigrateMemObjects({buffer_in35, buffer_in36}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q19.enqueueMigrateMemObjects({buffer_in37, buffer_in38}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q20.enqueueMigrateMemObjects({buffer_in39, buffer_in40}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q21.enqueueMigrateMemObjects({buffer_in41, buffer_in42}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q22.enqueueMigrateMemObjects({buffer_in43, buffer_in44}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q23.enqueueMigrateMemObjects({buffer_in45, buffer_in46}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q24.enqueueMigrateMemObjects({buffer_in47, buffer_in48}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q25.enqueueMigrateMemObjects({buffer_in49, buffer_in50}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q26.enqueueMigrateMemObjects({buffer_in51, buffer_in52}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q27.enqueueMigrateMemObjects({buffer_in53, buffer_in54}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q28.enqueueMigrateMemObjects({buffer_in55, buffer_in56}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q29.enqueueMigrateMemObjects({buffer_in57, buffer_in58}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q30.enqueueMigrateMemObjects({buffer_in59, buffer_in60}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q31.enqueueMigrateMemObjects({buffer_in61, buffer_in62}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q32.enqueueMigrateMemObjects({buffer_in63, buffer_in64}, 0 /* 0 means from host*/));


    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is
    // recommended
    // to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));
    OCL_CHECK(err, err = q2.enqueueTask(krnl_vector_add2));
    OCL_CHECK(err, err = q3.enqueueTask(krnl_vector_add3));
    OCL_CHECK(err, err = q4.enqueueTask(krnl_vector_add4));
    OCL_CHECK(err, err = q5.enqueueTask(krnl_vector_add5));
    OCL_CHECK(err, err = q6.enqueueTask(krnl_vector_add6));
    OCL_CHECK(err, err = q7.enqueueTask(krnl_vector_add7));
    OCL_CHECK(err, err = q8.enqueueTask(krnl_vector_add8));
    OCL_CHECK(err, err = q9.enqueueTask(krnl_vector_add9));
    OCL_CHECK(err, err = q10.enqueueTask(krnl_vector_add10));
    OCL_CHECK(err, err = q11.enqueueTask(krnl_vector_add11));
    OCL_CHECK(err, err = q12.enqueueTask(krnl_vector_add12));
    OCL_CHECK(err, err = q13.enqueueTask(krnl_vector_add13));
    OCL_CHECK(err, err = q14.enqueueTask(krnl_vector_add14));
    OCL_CHECK(err, err = q15.enqueueTask(krnl_vector_add15));
    OCL_CHECK(err, err = q16.enqueueTask(krnl_vector_add16));
    OCL_CHECK(err, err = q17.enqueueTask(krnl_vector_add17));
    OCL_CHECK(err, err = q18.enqueueTask(krnl_vector_add18));
    OCL_CHECK(err, err = q19.enqueueTask(krnl_vector_add19));
    OCL_CHECK(err, err = q20.enqueueTask(krnl_vector_add20));
    OCL_CHECK(err, err = q21.enqueueTask(krnl_vector_add21));
    OCL_CHECK(err, err = q22.enqueueTask(krnl_vector_add22));
    OCL_CHECK(err, err = q23.enqueueTask(krnl_vector_add23));
    OCL_CHECK(err, err = q24.enqueueTask(krnl_vector_add24));
    OCL_CHECK(err, err = q25.enqueueTask(krnl_vector_add25));
    OCL_CHECK(err, err = q26.enqueueTask(krnl_vector_add26));
    OCL_CHECK(err, err = q27.enqueueTask(krnl_vector_add27));
    OCL_CHECK(err, err = q28.enqueueTask(krnl_vector_add28));
    OCL_CHECK(err, err = q29.enqueueTask(krnl_vector_add29));
    OCL_CHECK(err, err = q30.enqueueTask(krnl_vector_add30));
    OCL_CHECK(err, err = q31.enqueueTask(krnl_vector_add31));
    OCL_CHECK(err, err = q32.enqueueTask(krnl_vector_add32));



    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in2}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q2.enqueueMigrateMemObjects({buffer_in4}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q3.enqueueMigrateMemObjects({buffer_in6}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q4.enqueueMigrateMemObjects({buffer_in8}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q5.enqueueMigrateMemObjects({buffer_in10}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q6.enqueueMigrateMemObjects({buffer_in12}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q7.enqueueMigrateMemObjects({buffer_in14}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q8.enqueueMigrateMemObjects({buffer_in16}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q9.enqueueMigrateMemObjects({buffer_in18}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q10.enqueueMigrateMemObjects({buffer_in20}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q11.enqueueMigrateMemObjects({buffer_in22}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q12.enqueueMigrateMemObjects({buffer_in24}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q13.enqueueMigrateMemObjects({buffer_in26}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q14.enqueueMigrateMemObjects({buffer_in28}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q15.enqueueMigrateMemObjects({buffer_in30}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q16.enqueueMigrateMemObjects({buffer_in32}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q17.enqueueMigrateMemObjects({buffer_in34}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q18.enqueueMigrateMemObjects({buffer_in36}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q19.enqueueMigrateMemObjects({buffer_in38}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q20.enqueueMigrateMemObjects({buffer_in40}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q21.enqueueMigrateMemObjects({buffer_in42}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q22.enqueueMigrateMemObjects({buffer_in44}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q23.enqueueMigrateMemObjects({buffer_in46}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q24.enqueueMigrateMemObjects({buffer_in48}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q25.enqueueMigrateMemObjects({buffer_in50}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q26.enqueueMigrateMemObjects({buffer_in52}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q27.enqueueMigrateMemObjects({buffer_in54}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q28.enqueueMigrateMemObjects({buffer_in56}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q29.enqueueMigrateMemObjects({buffer_in58}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q30.enqueueMigrateMemObjects({buffer_in60}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q31.enqueueMigrateMemObjects({buffer_in62}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q32.enqueueMigrateMemObjects({buffer_in64}, CL_MIGRATE_MEM_OBJECT_HOST));


    q.finish();
    q2.finish();
    q3.finish();
    q4.finish();
    q5.finish();
    q6.finish();
    q7.finish();
    q8.finish();
    q9.finish();
    q10.finish();
    q11.finish();
    q12.finish();
    q13.finish();
    q14.finish();
    q15.finish();
    q16.finish();
    q17.finish();
    q18.finish();
    q19.finish();
    q20.finish();
    q21.finish();
    q22.finish();
    q23.finish();
    q24.finish();
    q25.finish();
    q26.finish();
    q27.finish();
    q28.finish();
    q29.finish();
    q30.finish();
    q31.finish();
    q32.finish();

	for (int j=0; j<gsize; j++){
		ptrGray[j]=ptrg1[j];
		ptrGray[j+ gsize]=ptrg2[j];
		ptrGray[j+ 2*gsize]=ptrg3[j];
		ptrGray[j+3*gsize]=ptrg4[j];
		ptrGray[j+4*gsize]=ptrg5[j];
		ptrGray[j+5*gsize]=ptrg6[j];
		ptrGray[j+6*gsize]=ptrg7[j];
		ptrGray[j+7*gsize]=ptrg8[j];
		ptrGray[j+8*gsize]=ptrg9[j];
		ptrGray[j+ 9*gsize]=ptrg10[j];
		ptrGray[j+ 10*gsize]=ptrg11[j];
		ptrGray[j+11*gsize]=ptrg12[j];
		ptrGray[j+12*gsize]=ptrg13[j];
		ptrGray[j+13*gsize]=ptrg14[j];
		ptrGray[j+14*gsize]=ptrg15[j];
		ptrGray[j+15*gsize]=ptrg16[j];
		ptrGray[j+16*gsize]=ptrg17[j];
		ptrGray[j+ 17*gsize]=ptrg18[j];
		ptrGray[j+ 18*gsize]=ptrg19[j];
		ptrGray[j+19*gsize]=ptrg20[j];
		ptrGray[j+20*gsize]=ptrg21[j];
		ptrGray[j+21*gsize]=ptrg22[j];
		ptrGray[j+22*gsize]=ptrg23[j];
		ptrGray[j+23*gsize]=ptrg24[j];
		ptrGray[j+24*gsize]=ptrg25[j];
		ptrGray[j+ 25*gsize]=ptrg26[j];
		ptrGray[j+ 26*gsize]=ptrg27[j];
		ptrGray[j+27*gsize]=ptrg28[j];
		ptrGray[j+28*gsize]=ptrg29[j];
		ptrGray[j+29*gsize]=ptrg30[j];
		ptrGray[j+30*gsize]=ptrg31[j];
		if(j+31*gsize<imgRGB.width() * imgRGB.height()){
			ptrGray[j+31*gsize]=ptrg32[j];
		}
	}

    // OPENCL HOST CODE AREA END
    CImgDisplay dispGray(imgGrayScale, "FPGA Gray Scale");
    imgGrayScale.save_jpeg("/home/sureshm/Senior_Design_HPC/GPU/ImageProcessing/Gray_Scale/grey_mountain-landscape-reflection.jpg");
}
