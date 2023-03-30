#include "MatrixMult.h"
#include "GPUErrors.h"
__global__ void NaiveMult(float* g_A, float* g_B, float* g_C, const int ny, const int nx) {
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	float fSum = 0.0f;
	//This conditional is for debugging, even though done on the device

	if (row < ny && col < nx) { 
		for (int k = 0; k < nx; k++) {
			int idx_A = abs(row-k);
			int idx_B = abs(k-col);
			fSum += g_A[idx_A] * g_B[idx_B];
		}
		g_C[row * (nx) + col] = fSum;
	}
}



__host__ void gpuMultHelper(float* h_A, float* h_B, float* h_C, float* ref, const int ny, const int nx)
{
	float ElapsedTime{}, Elapsed_hd{}, Elapsed_dh{};
	float* d_A, * d_B, * d_C;
	const int MatrixSizeInBytes = nx * sizeof(float);//rowsxcolsxnumber of bytes
	const int C_size_in_Bytes=sizeof(float)*(nx*(ny));
	//Allocate device memory on the global memory
	cout<<(nx*(nx+1))/2<<endl;
	
	HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_B, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_C, C_size_in_Bytes));
    cudaEvent_t start, stop;
    cudaEvent_t start_dh, stop_dh;
    cudaEvent_t start_hd, stop_hd;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
    HandleCUDAError(cudaEventCreate(&start_dh));
    HandleCUDAError(cudaEventCreate(&stop_dh));
    HandleCUDAError(cudaEventCreate(&start_hd));
    HandleCUDAError(cudaEventCreate(&stop_hd));
    HandleCUDAError(cudaEventRecord(start_hd,0));
	//transfer data from CPU Memory to GPU Memory
	if(!HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice))){
		cout<<"cannot transfer A"<<endl;
	}

	if(!HandleCUDAError(cudaMemcpy(d_B, h_B, MatrixSizeInBytes, cudaMemcpyHostToDevice))){
		cout<<"cannot transfer B"<<endl;
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
	//Kernel Invoke Parameters - 2D Grid and 2D Blocks
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);//16x16 block
	//Never use the maximum number of threads in a block, the standard value is 256 or less
	dim3 grid((ny+block.y-1)/block.y, (nx + block.x - 1) / block.x);//First says the number of blocks in the y direction, this ensures number of blocks can handle data
	//We do not need to cast since the structure is with ints
	// cout << "\t2D Grid Dimension" << endl;
	// cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	// cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	// cout << "\t2D Block Dimension" << endl;
	// cout << "\tNumber of threads along X dimension: " << block.x << endl;
	// cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	//Launch Multiplication kernel
	HandleCUDAError(cudaEventRecord(start,0));
	NaiveMult << <grid, block >> > (d_A,d_B,d_C,ny,nx);
	cudaDeviceSynchronize();//Recall this makes the CPU wait
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
	//Copy product matrix to host
	if(!HandleCUDAError(cudaMemcpy(h_C, d_C, C_size_in_Bytes, cudaMemcpyDeviceToHost))){
		cout<<"Unable to transfer C"<<endl;
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
	
	//Verification code
	Verify(h_C,ref, ny);
	float total_time=Elapsed_dh+Elapsed_hd;
    float bytes_transferred=(2*MatrixSizeInBytes+C_size_in_Bytes)*1.0f;
    float throughput=(bytes_transferred*1e-6)/(total_time);
    cout<< "GPU MM Memory elapsed time: "<<total_time<< " ms"<<endl;
    cout<< "GPU MM Exec elapsed time: "<<ElapsedTime<< " ms"<<endl;
    cout<< "GPU MM total elapsed time: "<<ElapsedTime+total_time<< " ms"<<endl;
    cout<<"GPU throughput: "<<throughput<< "GB/s"<<endl;
	//Release Memory and reset device
	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_B));
	HandleCUDAError(cudaFree(d_C));

	cudaDeviceReset();
}