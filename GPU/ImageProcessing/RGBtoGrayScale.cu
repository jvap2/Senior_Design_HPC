#include "RGBtoGrayScale.h"
//Kernel Version 0
__global__ void gpu_RGBtoGrayScaleVer0(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w)
{
	unsigned int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < (h * w)) {
		out[idx] = .21f * in[idx] + .71f * in[idx + (h * w)] + .07 * in[idx + 2 * h * w];
	}
}
//Kernel Version 1
__global__ void gpu_RGBtoGrayScaleVer1(unsigned char* r,
	unsigned char* g,
	unsigned char* b,
	unsigned char* out, unsigned int h, unsigned int w)
{
	unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx < (h * w)) {
		out[idx] = 0.21f * *(r + idx) + 0.72f * *(g + idx) + .07f * *(b + idx);//This allows us to avoid multiplying 2 times offset
	}
}

//Kernel Version Two; 2D Implementation
__global__ void gpu_RGBtoGrayScaleVer2(unsigned char* r,
	unsigned char* g,
	unsigned char* b,
	unsigned char* out, unsigned int h, unsigned int w) {
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	//Compute the gray scale image index
	int idx = y * w+x;//y*w is row times width, so this points to the beginning of each row, x is the offset to get to a column
	if (x<w && y<h) {
		out[idx] = 0.21f * *(r + idx) + 0.72f * *(g + idx) + .07f * *(b + idx);//We can use this since we converted 2d to 1d
	}
	//When we are closer to the hardware we need to write this in 1D

}

//Host Helper function
__host__ double gpu_RGBtoGrayScaleHelper(unsigned char* h_in, unsigned char* h_out, 
	unsigned int rgbSIZE,
	unsigned int graySIZE,
	unsigned int h,
	unsigned int w,
	unsigned int kernelVer)
{
	double computeTime;
	unsigned char* d_in, * d_out;
	//Allocating device memory for the RGB and GrayScale Images
	if (!HandleCUDAError(cudaMalloc((void**) &d_in,rgbSIZE))) {
		cout << "Error allocating on the GPU for the RGB image" << endl;
		return FALSE;
	}
	//Allocate Memory on the GPU for Gray Scale
	if (!HandleCUDAError(cudaMalloc((void**)&d_out, graySIZE))) {
		cout << "Error allocating on the GPU for the gray scale image" << endl;
		return FALSE;
	}
	
	//Copying the RGB image to the device
	if (!HandleCUDAError(cudaMemcpy(d_in,h_in,rgbSIZE,cudaMemcpyHostToDevice))) {
		cout << "Error transferring RGB image from host to device" << endl;
		return FALSE;
	}
	//Setup Execution Configuration Parameters
	unsigned int threadsPerBlock = 256;//We are choosing 256 threads for 8 warps
	unsigned int blocksPerGrid = ((w * h) / threadsPerBlock)+1;//We are adding one since the the division is rounding down, and we need to account for extras
	
	cout << "Image Grid Size = " << (w * h) << " pixels" << endl;
	cout << "Number of threads per block = " << threadsPerBlock << endl;
	cout << "Number of blocks per Grid = " << blocksPerGrid << endl;
	cout << "Total Number of Threads in the Grid = " << threadsPerBlock * blocksPerGrid << endl;
	
	if (kernelVer == 0)
	{
		//Launch the RGB to Gray Scale Kernel - Ver 0
		auto start = high_resolution_clock::now();
		gpu_RGBtoGrayScaleVer0 << <blocksPerGrid, threadsPerBlock >> > (d_in,
			d_out,
			h,
			w);
		cudaDeviceSynchronize();
		auto end = high_resolution_clock::now();
		auto elasped_seconds = end - start;
		computeTime = duration_cast<microseconds>(elasped_seconds).count();
	}
	else if (kernelVer==1)
	{
		unsigned char* d_r = d_in;//We are pointing to the red
		unsigned char* d_g = d_in+(h*w);
		unsigned char* d_b = d_in + (2*h * w);
		//Launch the RGB to Gray Scale Kernel - Ver 1
		auto start = high_resolution_clock::now();//Comes from std::chrono namespace
		gpu_RGBtoGrayScaleVer1<< <blocksPerGrid, threadsPerBlock >> > (d_r,
			d_g,
			d_b,
			d_out,
			h,
			w);
		cudaDeviceSynchronize();
		auto end = high_resolution_clock::now();
		auto elasped_seconds = end - start;
		computeTime = duration_cast<microseconds>(elasped_seconds).count();

		
	}
	else {
		unsigned char* d_r = d_in;//We are pointing to the red
		unsigned char* d_g = d_in + (h * w);
		unsigned char* d_b = d_in + (2 * h * w);
		//Setup Execution Configuration Parameters
		int TILE_WIDTH = 16;
		dim3 dimGrid(ceil((float)w / TILE_WIDTH), ceil((float)h / TILE_WIDTH));
		dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);//Each block has 16x16 threads
		auto start = high_resolution_clock::now();
		gpu_RGBtoGrayScaleVer2 << <dimGrid, dimBlock >> > (d_r,
			d_g,
			d_b,
			d_out,
			h,
			w);
		cudaDeviceSynchronize();
		auto end = high_resolution_clock::now();
		auto elasped_seconds = end - start;
		computeTime = duration_cast<microseconds>(elasped_seconds).count();
	}
	//Copy the grayscale image data from device to host
	if (!HandleCUDAError(cudaMemcpy(h_out, d_out, graySIZE, cudaMemcpyDeviceToHost))) {
		cout << "Error transferring RGB image from host to device" << endl;
		return FALSE;
	}
	if (!HandleCUDAError(cudaFree(d_in)))
	{
		cout << "Error freeing RGB image memory" << endl;
		return FALSE;
	}
	if (!HandleCUDAError(cudaFree(d_out)))
	{
		cout << "Error freeing GrayScale image memory" << endl;
		return FALSE;
	}
	HandleCUDAError(cudaDeviceReset());
	return computeTime;
}