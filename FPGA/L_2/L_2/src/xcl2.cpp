#include "xcl2.hpp"
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

namespace xcl {
std::vector<cl::Device> get_devices(const std::string &vendor_name) {
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++) {
        platform = platforms[i];
        OCL_CHECK(err,
                  std::string platformName =
                      platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (platformName == vendor_name) {
            std::cout << "Found Platform" << std::endl;
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }
    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    OCL_CHECK(err,
              err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}

std::vector<cl::Device> get_xil_devices() { return get_devices("Xilinx"); }

std::vector<unsigned char>
read_binary_file(const std::string &xclbin_file_name) {
    std::cout << "INFO: Reading " << xclbin_file_name << std::endl;

    if (access(xclbin_file_name.c_str(), R_OK) != 0) {
        printf("ERROR: %s xclbin not available please build\n",
               xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    std::cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<unsigned char> buf;
    buf.resize(nb);
    bin_file.read(reinterpret_cast<char *>(buf.data()), nb);
    return buf;
}

bool is_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if (xcl_mode != NULL) {
        ret = true;
    }
    return ret;
}

bool is_hw_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if ((xcl_mode != NULL) && !strcmp(xcl_mode, "hw_emu")) {
        ret = true;
    }
    return ret;
}

bool is_xpr_device(const char *device_name) {
    const char *output = strstr(device_name, "xpr");

    if (output == NULL) {
        return false;
    } else {
        return true;
    }
}
}; // namespace xcl

void fpga_init(cl::Context *context, cl::CommandQueue *queue, cl::Kernel *kernel, std::string fpga_bitstream, const char *kernel_name, bool verbose){

    cl_int err;
    auto devices = xcl::get_xil_devices();
    if (verbose) std::cout << "Reading FPGA bitstream..." << std::endl;
    auto bistream_file = xcl::read_binary_file(fpga_bitstream);
    cl::Program::Binaries bins{{bistream_file.data(), bistream_file.size()}};

    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++){
        auto device = devices[i];

        OCL_CHECK(err, *context = cl::Context({device}, NULL, NULL, NULL, &err));
        OCL_CHECK(err, *queue = cl::CommandQueue(*context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));

        if (verbose) std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        OCL_CHECK(err, cl::Program program(*context, {device}, bins, NULL, &err));
        if (err != CL_SUCCESS){
            if (verbose) std::cout << "Failed to program device[" << i << "] with xclbin file!" << std::endl;
        }else{
            if (verbose) std::cout << "Device[" << i << "]: program successful!" << std::endl;
            OCL_CHECK(err, *kernel = cl::Kernel(program, kernel_name, &err));
            valid_device++;
            break; // Break because a valid device was found
        }
    }
    if (valid_device == 0){
        std::cout << "Failed to program any device found!" << std::endl;
        exit(EXIT_FAILURE);
    }
}
