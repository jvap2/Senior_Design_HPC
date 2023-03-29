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
	float* d_A, * d_B, * d_C;
	const int MatrixSizeInBytes = nx * sizeof(float);//rowsxcolsxnumber of bytes
	const int C_size_in_Bytes=sizeof(float)*(nx*(ny));
	//Allocate device memory on the global memory
	cout<<(nx*(nx+1))/2<<endl;
	
	HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_B, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_C, C_size_in_Bytes));


	//transfer data from CPU Memory to GPU Memory
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	if(!HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice))){
		cout<<"cannot transfer A"<<endl;
	}

	if(!HandleCUDAError(cudaMemcpy(d_B, h_B, MatrixSizeInBytes, cudaMemcpyHostToDevice))){
		cout<<"cannot transfer B"<<endl;
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout << "GPU Memory Transfer time (H to D): " << (elapsed_seconds.count() * 1000.0f) << " msecs" << endl;

	//Kernel Invoke Parameters - 2D Grid and 2D Blocks
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);//16x16 block
	//Never use the maximum number of threads in a block, the standard value is 256 or less
	dim3 grid((ny+block.y-1)/block.y, (nx + block.x - 1) / block.x);//First says the number of blocks in the y direction, this ensures number of blocks can handle data
	//We do not need to cast since the structure is with ints
	cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	//Launch Multiplication kernel
	start = std::chrono::system_clock::now();
	NaiveMult << <grid, block >> > (d_A,d_B,d_C,ny,nx);
	cudaDeviceSynchronize();//Recall this makes the CPU wait
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	cout << "Naive Multiplication: GPU Execution time: " << (elapsed_seconds.count() * 1000.0f) << " msecs" << endl;

	start = std::chrono::system_clock::now();
	//Copy product matrix to host
	if(!HandleCUDAError(cudaMemcpy(h_C, d_C, C_size_in_Bytes, cudaMemcpyDeviceToHost))){
		cout<<"Unable to transfer C"<<endl;
	}
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	cout << "GPU Memory Transfer time (D to H): " << (elapsed_seconds.count() * 1000.0f) << " msecs" << endl;
	
	//Verification code
	Verify(h_C,ref, ny);
	
	//Release Memory and reset device
	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_B));
	HandleCUDAError(cudaFree(d_C));

	cudaDeviceReset();
}