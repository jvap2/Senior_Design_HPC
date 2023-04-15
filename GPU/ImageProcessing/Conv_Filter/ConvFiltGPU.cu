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
#define TILE_WIDTH 16





__host__ void Helper_Filter(unsigned char* h_in, unsigned char* h_out,unsigned int greySIZE,
	unsigned int blurSIZE,
	unsigned int h,
	unsigned int w){
    unsigned char* d_in, * d_out;
	float ElapsedTime{}, Elapsed_hd{}, Elapsed_dh{};
	//Allocating device memory for the RGB and GrayScale Images
	cudaEvent_t start, stop;
    cudaEvent_t start_dh, stop_dh;
    cudaEvent_t start_hd, stop_hd;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
    HandleCUDAError(cudaEventCreate(&start_dh));
    HandleCUDAError(cudaEventCreate(&stop_dh));
    HandleCUDAError(cudaEventCreate(&start_hd));
    HandleCUDAError(cudaEventCreate(&stop_hd));
	if (!HandleCUDAError(cudaMalloc((void**) &d_in,greySIZE))) {
	}
	//Allocate Memory on the GPU for Gray Scale
	if (!HandleCUDAError(cudaMalloc((void**)&d_out, blurSIZE))) {
		cout << "Error allocating on the GPU for the gray scale image" << endl;
	}
	HandleCUDAError(cudaEventRecord(start_hd,0));
	//Copying the RGB image to the device
	if (!HandleCUDAError(cudaMemcpy(d_in,h_in,greySIZE,cudaMemcpyHostToDevice))) {
		cout << "Error transferring RGB image from host to device" << endl;
	}
	if (!HandleCUDAError(cudaEventRecord(stop_hd, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop_hd))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Save the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&Elapsed_hd, start_hd, stop_hd))) {
		cout << "Unable to find elapsed time between events" << endl;
	}

	dim3 dimGrid(ceil((float)w / TILE_WIDTH), ceil((float)h / TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << dimGrid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << dimGrid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << dimBlock.x << endl;
	cout << "\tNumber of threads along Y dimension: " << dimBlock.y << endl;
	HandleCUDAError(cudaEventRecord(start,0));
    d_Gauss_Filter<<<dimGrid,dimBlock>>>(d_in,d_out,h,w);
	if (!HandleCUDAError(cudaEventRecord(stop, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Save the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	HandleCUDAError(cudaEventRecord(start_dh,0));
	if (!HandleCUDAError(cudaMemcpy(h_out, d_out, blurSIZE, cudaMemcpyDeviceToHost))) {
		cout << "Error transferring RGB image from host to device" << endl;
	}
	if (!HandleCUDAError(cudaEventRecord(stop_dh, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop_dh))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Save the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&Elapsed_dh, start_dh, stop_dh))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
    if (!HandleCUDAError(cudaFree(d_in)))
	{
		cout << "Error freeing RGB image memory" << endl;
	}
	if (!HandleCUDAError(cudaFree(d_out)))
	{
		cout << "Error freeing GrayScale image memory" << endl;
	}
	float total_time=Elapsed_dh+Elapsed_hd;
    float bytes_transferred=(greySIZE+blurSIZE)*1.0f;
    float throughput=(bytes_transferred*1e-6)/(total_time);
    cout<< "GPU Gauss Filter Memory elapsed time: "<<total_time<< " ms"<<endl;
    cout<< "GPU Gauss Filter elapsed time: "<<ElapsedTime<< " ms"<<endl;
    cout<< "GPU Gauss Filter total elapsed time: "<<ElapsedTime+total_time<< " ms"<<endl;
    cout<<"GPU throughput: "<<throughput<< "GB/s"<<endl;
	HandleCUDAError(cudaDeviceReset());

}
