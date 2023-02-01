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
	float ElapsedTime_dh{};
	float ElapsedTime_hd{};
    cudaEvent_t start, stop;
    cudaEvent_t start_dh, stop_dh;
    cudaEvent_t start_hd, stop_hd;
    int t_B=256;
    int b_G=ceil(1.0*size/t_B);
    HandleCUDAError(cudaEventCreate(&start));
    HandleCUDAError(cudaEventCreate(&stop));
    HandleCUDAError(cudaEventCreate(&start_dh));
    HandleCUDAError(cudaEventCreate(&stop_dh));
    HandleCUDAError(cudaEventCreate(&start_hd));
    HandleCUDAError(cudaEventCreate(&stop_hd));
    HandleCUDAError(cudaMalloc((void**)&d_A,SizeInBytes));
    HandleCUDAError(cudaMalloc((void**)&d_B,SizeInBytes));
    HandleCUDAError(cudaMalloc((void**)&d_C,SizeInBytes));

    HandleCUDAError(cudaEventRecord(start_hd,0));
    HandleCUDAError(cudaMemcpy(d_A,h_A,SizeInBytes,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_B,h_B,SizeInBytes,cudaMemcpyHostToDevice));
    if (!HandleCUDAError(cudaEventRecord(stop_hd, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop_hd))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Save the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime_hd, start_hd, stop_hd))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
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
    HandleCUDAError(cudaEventRecord(start_dh,0));
    HandleCUDAError(cudaMemcpy(h_C,d_C,SizeInBytes,cudaMemcpyDeviceToHost));
    if (!HandleCUDAError(cudaEventRecord(stop_dh, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop_dh))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Save the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime_dh, start_dh, stop_dh))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
    cout<<"Memcpy time: "<<ElapsedTime_dh+ElapsedTime_hd<<" ms"<<endl;
    VaddVerification(ref,h_C,size);
    HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));
	HandleCUDAError(cudaEventDestroy(start_dh));
	HandleCUDAError(cudaEventDestroy(stop_dh));
	HandleCUDAError(cudaEventDestroy(start_hd));
	HandleCUDAError(cudaEventDestroy(stop_hd));
    HandleCUDAError(cudaFree(d_A));
    HandleCUDAError(cudaFree(d_B));
    HandleCUDAError(cudaFree(d_C));
    HandleCUDAError(cudaDeviceReset());
}