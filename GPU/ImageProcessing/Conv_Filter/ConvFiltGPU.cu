#include "GPUErrors.h"
#include "ConvFilt.h"


__global__ void d_Filter_v0(unsigned char* in, unsigned char* out,int h, int w, int horiz_off, int vert_off){
    int x=threadIdx.x+(blockIdx.x*blockDim.x);
    int y=threadIdx.y+(blockIdx.y*blockDim.y);
    int sum_r{};
    int sum_g{};
    int sum_b{};
    int stride=h*w;
    int stride_2=2*h*w;
    int size_filt=vert_off*horiz_off;
    if(x < (w) && y<(h)){
        for(int y_off{}; y_off<vert_off;y_off++){
            int row=y+y_off-(vert_off/2);
            for(int x_off{}; x_off<horiz_off; x_off++){
                int col=x+x_off-(horiz_off/2);
                if(col>=0 && row>=0 && row<h && col<w){
                    sum_r+=in[row*w+col];
                    sum_g+=in[row*w+col+stride];
                    sum_b+=in[row*w+col+stride_2];
                }
            }
        }
        out[y*w+x]=(char)sum_r/size_filt;
        out[y*w+x+stride]=(char)sum_g/size_filt;
        out[y*w+x+stride_2]=(char)sum_b/size_filt;
    }
}


__host__ void Helper_Filter(unsigned char* h_in, unsigned char* h_out,	unsigned int rgbSIZE,
	unsigned int blurSIZE,
	unsigned int h,
	unsigned int w,
    unsigned int h_off,
    unsigned int w_off){
    unsigned char* d_in, * d_out;
	//Allocating device memory for the RGB and GrayScale Images
	if (!HandleCUDAError(cudaMalloc((void**) &d_in,rgbSIZE))) {
	}
	//Allocate Memory on the GPU for Gray Scale
	if (!HandleCUDAError(cudaMalloc((void**)&d_out, blurSIZE))) {
		cout << "Error allocating on the GPU for the gray scale image" << endl;
	}
	
	//Copying the RGB image to the device
	if (!HandleCUDAError(cudaMemcpy(d_in,h_in,rgbSIZE,cudaMemcpyHostToDevice))) {
		cout << "Error transferring RGB image from host to device" << endl;
	}

	int dimx = 16;
	int dimy = 16;

	dim3 block(dimx, dimy);
	dim3 grid((w + block.y - 1) / block.y, (h + block.x - 1) / block.x);
    cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;
    d_Filter_v0<<<grid,block>>>(d_in,d_out,h,w,w_off,h_off);
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