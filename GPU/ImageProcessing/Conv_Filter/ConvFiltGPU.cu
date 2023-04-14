#include "GPUErrors.h"
#include "ConvFilt.h"


__global__ void d_Gauss_Filter(unsigned char* in, unsigned char* out,int h, int w){
	// __shared__ unsigned char in_shared[16][16];
    int x=threadIdx.x+(blockIdx.x*blockDim.x);
    int y=threadIdx.y+(blockIdx.y*blockDim.y);
	int idx = y * (w-2)+x;
    if(x < (w-2) && y<(h-2)){
            out[idx]+=.0625*in[y*w+x];
            out[idx]+=.125*in[y*w+x+1];
            out[idx]+=.0625*in[y*w+x+2];
            out[idx]+=.125*in[(y+1)*w+x];
            out[idx]+=.25*in[(y+1)*w+x+1];
            out[idx]+=.125*in[(y+1)*w+x+2];
            out[idx]+=.0625*in[(y+2)*w+x];
            out[idx]+=.125*in[(y+2)*w+x+1];
            out[idx]+=.0625*in[(y+2)*w+x+2];
    }
}




__host__ void Helper_Filter(unsigned char* h_in, unsigned char* h_out,unsigned int greySIZE,
	unsigned int blurSIZE,
	unsigned int h,
	unsigned int w){
    unsigned char* d_in, * d_out;
	//Allocating device memory for the RGB and GrayScale Images
	if (!HandleCUDAError(cudaMalloc((void**) &d_in,greySIZE))) {
	}
	//Allocate Memory on the GPU for Gray Scale
	if (!HandleCUDAError(cudaMalloc((void**)&d_out, blurSIZE))) {
		cout << "Error allocating on the GPU for the gray scale image" << endl;
	}
	
	//Copying the RGB image to the device
	if (!HandleCUDAError(cudaMemcpy(d_in,h_in,greySIZE,cudaMemcpyHostToDevice))) {
		cout << "Error transferring RGB image from host to device" << endl;
	}

	int TILE_WIDTH = 16;
	dim3 dimGrid(ceil((float)w / TILE_WIDTH), ceil((float)h / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << dimGrid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << dimGrid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << dimBlock.x << endl;
	cout << "\tNumber of threads along Y dimension: " << dimBlock.y << endl;
    d_Gauss_Filter<<<dimGrid,dimBlock>>>(d_in,d_out,h,w);
	if (!HandleCUDAError(cudaMemcpy(h_out, d_out, blurSIZE, cudaMemcpyDeviceToHost))) {
		cout << "Error transferring RGB image from host to device" << endl;
	}
    if (!HandleCUDAError(cudaFree(d_in)))
	{
		cout << "Error freeing RGB image memory" << endl;
	}
	if (!HandleCUDAError(cudaFree(d_out)))
	{
		cout << "Error freeing GrayScale image memory" << endl;
	}
	HandleCUDAError(cudaDeviceReset());

}
