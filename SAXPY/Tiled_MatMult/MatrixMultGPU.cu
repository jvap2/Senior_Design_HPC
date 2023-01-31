#include "MatrixMult.h"
#include "GPUErrors.h"

__global__ void NaiveMult(float* g_A, float* g_B, float* g_C, const int ny, const int nx)
{
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	float fSum = 0.0f;
	if (row < ny && col < nx)
	{
		for (int k = 0; k < nx; k++)
		{
			fSum += g_A[row * nx + k] * g_B[k * nx + col];
		}
		g_C[row * nx + col] = fSum;
	}
}

//Tiled Kernel
#define TILE_WIDTH 32
__global__ void TiledMult(float* g_A, float* g_B, float* g_C, const int Width)
{
	//Define a static 2D array on the shared memory of size TILE_WIDTH * TILE_WIDTH to store the elements of the matrix A
	//We do not need to access the static array with one index, memory is stored as matrix and since static. If it is dynamic, we have to access with 1D
	__shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
	//Define a static 2D array on the shared memory of size TILE_WIDTH * TILE_WIDTH to store the elements of the matrix B
	//TILE_WIDTH is equal to block width since this is square matrices
	__shared__ float Bds[TILE_WIDTH][TILE_WIDTH];//Ads and Bds are the portions of data we are going to allocate into the shared mem
	//Shared memory is only around for the lifetime of a block. Once evicted, shared memory is gone
	// If we have 4 blocks running on an SM, there are 4 copies of Ads and Bds on SM
	//Write code to store locally the thread and block indices
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	//This is for ease of typing these- we are wasting registers, but for this problem it won't matter
	//Compute the row and column indices of the C Element to store the dot product result
	int row = ty + (by * TILE_WIDTH);
	int col = tx + (bx * TILE_WIDTH);//Note the difference, the width here is tile_width. In the naive, we use blockDim as the width

	//Loop over the A and B tiles to compute the C Element
	float cValue = 0.0f;
	for (int ph{}; ph < Width / TILE_WIDTH;++ph) {
		//Load A and B tiles into the shared memory collaboratively
		//Note, we are shifting by ph*TILE_WIDTH so when the next phase happens, we can get one TILE_WIDTH away
		Ads[ty][tx] = g_A[row * Width + ph * TILE_WIDTH + tx];//Add tx to get the column in a tile
		Bds[ty][tx] = g_B[(ph * TILE_WIDTH + ty) * Width+col];//The term in () shifts us down the column, we are using column to shift across the tile horizontally
		//One thread gets one element
		//col is within the size of the tile width, so we will stay within a tile as this increments
		//This is called dynamic core model
		//Wait for threads of the block to fetch their specific element of block (TILE_WIDTH) to complete loading to shared memory
		__syncthreads(); 
		//Perform the partial dot product in the phase
		for (int k{}; k < TILE_WIDTH;k++) {
			cValue += Ads[ty][k] * Bds[k][tx];
			//We access A in a coalesced access in the shared memory
			//We are only doing this across the tile
		}
		//Wait for all threads in the block to complete partial dot product in a phase
		__syncthreads();
	}
	//We have now finished the dot product
	g_C[row * Width + col] = cValue;
}

__host__ void gpuMultHelper(float* h_A, float* h_B, float* h_C, float* h_C_Tile,float* ref, const int ny, const int nx)
{
	float* d_A, * d_B, * d_C;
	const int MatrixSizeInBytes = ny * nx * sizeof(float);
	float ElapsedTime{};
	// chrono::time_point<high_resolution_clock> start, end;
	// double computeTime{};

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_B, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_C, MatrixSizeInBytes));
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
	//transfer data from CPU Memory to GPU Memory
	HandleCUDAError(cudaEventRecord(start,0));
	HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice));
	HandleCUDAError(cudaMemcpy(d_B, h_B, MatrixSizeInBytes, cudaMemcpyHostToDevice));
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
	cout<< "Naive Mat Mult Memcpy H->D: "<<ElapsedTime<< " ms"<<endl;
	//Kernel Invoke Parameters - 2D Grid and 2D Blocks
	int dimx = 32;
	int dimy = 32;

	dim3 block(dimy, dimx);
	dim3 grid((ny + block.y - 1) / block.y, (nx + block.x - 1) / block.x);

	cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	//Executing Naive Multiplication
	HandleCUDAError(cudaEventRecord(start,0));
	NaiveMult << <grid, block >> > (d_A, d_B, d_C, nx, ny);
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
	cout << "Naive Multiplication: GPU Execution time: " << ElapsedTime << " msecs" << endl;
	HandleCUDAError(cudaEventRecord(start,0));
	HandleCUDAError(cudaMemcpy(h_C, d_C, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
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
	cout<< "Naive Mat Mult Memcpy D->H: "<<ElapsedTime<< " msecs" <<endl;
	MatrixMultVerification(ref, h_C, ny, nx);
	//Release the device memory of the C Matrix
	HandleCUDAError(cudaFree(d_C));//We are releasing the memory of the product matrix so there is no effect of stack or cache
	
	//Executing the Tiled Matrix Multiplication
	//Reallocate the device memory of the C Matrix
	HandleCUDAError(cudaMalloc((void**)&d_C, MatrixSizeInBytes));
	// HandleCUDAError(cudaMalloc((void**)&d_C, MatrixSizeInBytes));
	HandleCUDAError(cudaEventRecord(start,0));
	TiledMult << <grid, block >> > (d_A, d_B, d_C, nx);//If we had third parameter, it would be for dynamically allocated shared memory
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
	cout << "Tiled Multiplication: GPU Execution time: " << ElapsedTime << " msecs" << endl;


	HandleCUDAError(cudaMemcpy(h_C_Tile, d_C, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	MatrixMultVerification(ref, h_C_Tile, ny, nx);
	HandleCUDAError(cudaFree(d_C));
	
	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_B));
	HandleCUDAError(cudaDeviceReset());
}