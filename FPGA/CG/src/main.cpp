#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include "CG.h"
#define DATA_SIZE 1024
#define ny 1024
#define nx 1024
#define MAX_HBM_BANKCOUNT 32
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string binaryFile = argv[1];
    size_t vector_size_bytes = sizeof(float) * DATA_SIZE;
    size_t Mat_Size_bytes=sizeof(float)*DATA_SIZE*DATA_SIZE;
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
    const int bank[MAX_HBM_BANKCOUNT] = {
        BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
        BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
        BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
        BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
        BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
        BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
        BANK_NAME(30), BANK_NAME(31)};

    float lambda[1];
    float beta[1];
    float bet{};
    float l{};
//    int* b{};
//    int* b_res{};
//    int* x{};
//    int* x_old{};
    int flag[1];
    int iter[1];
    // Create the test data
    float r[DATA_SIZE]{};
    float r_old[DATA_SIZE]{};
    float d[DATA_SIZE]{};
    float d_old[DATA_SIZE]{};
    float x[DATA_SIZE]{};
    float x_old[DATA_SIZE]{};
    float r_FPGA[DATA_SIZE]{};
    float r_old_FPGA[DATA_SIZE]{};
    float d_FPGA[DATA_SIZE]{};
    float d_old_FPGA[DATA_SIZE]{};
    float x_FPGA[DATA_SIZE]{};
    float x_old_FPGA[DATA_SIZE]{};
    float A[DATA_SIZE*DATA_SIZE]{};
    float A_T[DATA_SIZE*DATA_SIZE]{};
    float A_fin[DATA_SIZE*DATA_SIZE]{};
    float Ax[DATA_SIZE]{};
    float b[DATA_SIZE]{};
    float b_res[DATA_SIZE]{};
    *flag=1;
    *iter=1;
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

    for(int i=0; i<ny;i++){
    	r_FPGA[i]=0.0f;
    	d_FPGA[i]=0.0f;
    	x_FPGA[i]=0.0f;
        x_old_FPGA[i]=x_old[i];
        r_old_FPGA[i]=r_old[i];
        d_old[i]=r_old[i];
        d_old_FPGA[i]=d_old[i];
    }
    chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    C_G(A_fin,r,r_old,d,d_old,x,x_old,&bet,&l,ny,iter);
    end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	cout<<"Iterations: "<<*iter<<endl;
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
    //Create Pointers to go to other HBM banks
    cl_mem_ext_ptr_t A_cl,r_cl,r_old_cl,d_cl,d_old_cl,x_cl,x_old_cl;
    A_cl.obj=A_fin;
    A_cl.param=0;
    A_cl.flags=bank[1];
    r_cl.obj=r_FPGA;
    r_cl.param=0;
    r_cl.flags=bank[2];
    r_old_cl.obj=r_old_FPGA;
    r_old_cl.param=0;
    r_old_cl.flags=bank[2];
    d_cl.obj=d_FPGA;
    d_cl.param=0;
    d_cl.flags=bank[3];
    d_old_cl.obj=d_old_FPGA;
    d_old_cl.param=0;
    d_old_cl.flags=bank[3];
    x_cl.obj=x_FPGA;
    x_cl.param=0;
    x_cl.flags=bank[4];
    x_old_cl.obj=x_old_FPGA;
    x_old_cl.param=0;
    x_old_cl.flags=bank[4];


    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX ,Mat_Size_bytes,
                                         &A_cl, &err));
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX , vector_size_bytes,
    									&r_cl, &err));
    OCL_CHECK(err, cl::Buffer buffer_in3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, vector_size_bytes,
                                         &r_old_cl, &err));
    OCL_CHECK(err, cl::Buffer buffer_in4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, vector_size_bytes,
                                         &d_cl, &err));
    OCL_CHECK(err, cl::Buffer buffer_in5(context, CL_MEM_USE_HOST_PTR |CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, vector_size_bytes,
                                         &d_old_cl, &err));
    OCL_CHECK(err, cl::Buffer buffer_in6(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, vector_size_bytes,
                                         &x_cl, &err));
    OCL_CHECK(err, cl::Buffer buffer_in7(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, vector_size_bytes,
                                         &x_old_cl, &err));
    OCL_CHECK(err, cl::Buffer BET(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (sizeof(float)),
                                            (void*)(beta), &err));
    OCL_CHECK(err, cl::Buffer lam(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (sizeof(float)),
                                            (void*)(lambda), &err));
    OCL_CHECK(err, cl::Buffer it(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (sizeof(float)),
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
    int i=0;
    Verify(x_FPGA,x,ny,&match,&i);
    if(match==false){
    	std::cout << "Error: Result mismatch" << std::endl;
    	std::cout << "i = " << i << " CPU result = " << x[i]<< " Device result = " << x_FPGA[i] << std::endl;
    	match = false;
    }
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
