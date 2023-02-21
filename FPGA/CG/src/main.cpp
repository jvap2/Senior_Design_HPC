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
#include <iostream>
#include <cmath>
#include "CG.hpp"
#define DATA_SIZE 64
#define ny 64
#define nx 64


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string binaryFile = argv[1];
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    size_t Mat_Size_bytes=sizeof(int)*DATA_SIZE*DATA_SIZE;
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_vector_add;
    cl::CommandQueue q;
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

//    int* A{};
//    int* A_T{};
//    int* A_fin{};
//    int* r{};
//    int* r_old{};
//    int* d{};
//    int* d_old{};
    int lambda[1];
    int beta[1];
    int bet{};
    int l{};
//    int* b{};
//    int* b_res{};
//    int* x{};
//    int* x_old{};
    int flag[1];
    int iter[1]{};
    // Create the test data
    int r[DATA_SIZE]{};
    int r_old[DATA_SIZE]{};
    int d[DATA_SIZE]{};
    int d_old[DATA_SIZE]{};
    int x[DATA_SIZE]{};
    int x_old[DATA_SIZE]{};
    int r_FPGA[DATA_SIZE]{};
    int r_old_FPGA[DATA_SIZE]{};
    int d_FPGA[DATA_SIZE]{};
    int d_old_FPGA[DATA_SIZE]{};
    int x_FPGA[DATA_SIZE]{};
    int x_old_FPGA[DATA_SIZE]{};
    int A[DATA_SIZE*DATA_SIZE]{};
    int A_T[DATA_SIZE*DATA_SIZE]{};
    int A_fin[DATA_SIZE*DATA_SIZE]{};
    int Ax[DATA_SIZE]{};
    int b[DATA_SIZE]{};
    int b_res[DATA_SIZE]{};
    *flag=1;
//    for (int i = 0; i < DATA_SIZE; i++) {
//    	source_in1[i]=(int)(rand())/(int)(RAND_MAX);
//    	source_in2[i]=(int)(rand())/(int)(RAND_MAX);
//    	std::cout<<source_in1[i]<<std::endl;
//        source_sw_results += source_in1[i] * source_in2[i];
//    }
    InitializeMatrix(A,ny,nx);
    Diag_Dominant_Opt(A,ny);
    TransposeOnCPU(A,A_T,ny,ny);
    Generate_Vector(b,ny);
    cpuMatrixMult(A,A_T,A_fin,ny,nx);
    cpuMatrixVect(A_T,b,b_res,ny,nx);
    Generate_Vector(x_old,ny);
    cpuMatrixVect(A_fin, x_old, Ax, ny, nx);
    vector_subtract(b_res,Ax,r_old,ny);
    vector_subtract(b_res,Ax,r_old_FPGA,ny);

    for(int i=0; i<ny;i++){
        x_old_FPGA[i]=x_old[i];
        d_old[i]=r_old[i];
        d_old_FPGA[i]=d_old[i];
    }
    chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    C_G(A_fin,r,r_old,d,d_old,x,x_old,bet,l,ny,iter);
    end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;

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
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "final", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    //Need to calculate r and d


    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY ,Mat_Size_bytes,
                                         (void*)A_fin, &err));
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE , vector_size_bytes,
                                         (void*)r_FPGA, &err));
    OCL_CHECK(err, cl::Buffer buffer_in3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE , vector_size_bytes,
                                         (void*)r_old_FPGA, &err));
    OCL_CHECK(err, cl::Buffer buffer_in4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE , vector_size_bytes,
                                         (void*)d_FPGA, &err));
    OCL_CHECK(err, cl::Buffer buffer_in5(context, CL_MEM_USE_HOST_PTR |CL_MEM_READ_WRITE , vector_size_bytes,
                                         (void*)d_old_FPGA, &err));
    OCL_CHECK(err, cl::Buffer buffer_in6(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, vector_size_bytes,
                                         (void*)x_FPGA, &err));
    OCL_CHECK(err, cl::Buffer buffer_in7(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE , vector_size_bytes,
                                         (void*)x_old_FPGA, &err));
    OCL_CHECK(err, cl::Buffer BET(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (sizeof(int)),
                                            (void*)(beta), &err));
    OCL_CHECK(err, cl::Buffer lam(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (sizeof(int)),
                                            (void*)(lambda), &err));
    OCL_CHECK(err, cl::Buffer it(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (sizeof(int)),
                                            (void*)(iter), &err));

//    int size = DATA_SIZE;
    OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_in1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, buffer_in2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, buffer_in3));
    OCL_CHECK(err, err = krnl_vector_add.setArg(3, buffer_in4));
    OCL_CHECK(err, err = krnl_vector_add.setArg(4, buffer_in5));
    OCL_CHECK(err, err = krnl_vector_add.setArg(5, buffer_in6));
    OCL_CHECK(err, err = krnl_vector_add.setArg(6, buffer_in7));
    OCL_CHECK(err, err = krnl_vector_add.setArg(7, BET));
    OCL_CHECK(err, err = krnl_vector_add.setArg(8, lam));
    OCL_CHECK(err, err = krnl_vector_add.setArg(9, it));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2,buffer_in3, buffer_in4,buffer_in5, buffer_in6,buffer_in7,BET,lam,it}, 0 /* 0 means from host*/));

    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is
    // recommended
    // to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in6}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    bool match = true;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (x_FPGA[i] != x[i]) {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << x[i]
                      << " Device result = " << x_FPGA[i] << std::endl;
            match = false;
            break;
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
