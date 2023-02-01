#include "GPUErrors.h"
#include "Vadd.h"

__global__ void VectAdd(float* g_A, float* g_B, float* g_C, int size){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x);
    if(idx<size){
        g_C[idx]=g_A[idx]+g_B[idx];
    }
}

__host__ void gpuVaddHelper(float* h_A, float* h_B, float* h_C, float* ref, const int size){
    float* d_A, * d_B, * d_C;
	const int SizeInBytes = size* sizeof(float);
	float ElapsedTime{};
    cudaEvent_t start, stop;
    int t_B=256;
    int b_G=ceil(1.0*size/t_B);
    HandleCUDAError(cudaEventCreate(&start));
    HandleCUDAError(cudaEventCreate(&stop));
    HandleCUDAError(cudaMalloc((void**)&d_A,SizeInBytes));
    HandleCUDAError(cudaMalloc((void**)&d_B,SizeInBytes));
    HandleCUDAError(cudaMalloc((void**)&d_C,SizeInBytes));

    HandleCUDAError(cudaEventRecord(start,0));
    HandleCUDAError(cudaMemcpy(d_A,h_A,SizeInBytes,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_B,h_B,SizeInBytes,cudaMemcpyHostToDevice));
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
    cout<<"Memcpy from H=>D: "<<ElapsedTime<<" ms"<<endl;
    HandleCUDAError(cudaEventRecord(start,0));
    VectAdd<<<b_G,t_B>>>(d_A,d_B,d_C,size);
    cudaDeviceSynchronize();
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
    cout<<"GPU Vadd: "<<ElapsedTime<<" ms"<<endl;
    HandleCUDAError(cudaEventRecord(start,0));
    HandleCUDAError(cudaMemcpy(h_C,d_C,SizeInBytes,cudaMemcpyDeviceToHost));
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
    cout<<"Memcpy from D=>H: "<<ElapsedTime<<" ms"<<endl;
    VaddVerification(ref,h_C,size);
    HandleCUDAError(cudaFree(d_A));
    HandleCUDAError(cudaFree(d_B));
    HandleCUDAError(cudaFree(d_C));
    HandleCUDAError(cudaDeviceReset());
}