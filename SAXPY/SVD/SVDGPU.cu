#include "SVD.h"
#include "GPUErrors.h"

#define BX 16
#define BY 16

__global__ void Outer_Product(float* w, float* v, float* out, int k, int ny, int nx){
    int row=threadIdx.y+(blockDim.y*blockIdx.y);
    int col=threadIdx.x+(blockDim.x*blockIdx.x);
    out[row*ny+col]=w[row]*v[col];
}

__global__ void Dot_Product(float* w, float* v, float* hold, float out, int k, int size){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x);
    int tid=threadIdx.x;
    hold[idx]=w[idx]*v[idx];
    __syncthreads();
    float* blockAddress = hold + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block

	if (idx >= size)
	{
		return;//Leave the kernel if the global index of our thread exceeds the size of the vector being processed
	}
	//Perform the interleaved reduction, used to reduce divergence.
	//Start adding elements blockDim.x apart, store in place and then half the stride and continue until stride=1
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride && tid + stride < size)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
	if (tid == 0)
	{
		hold[blockIdx.x] = blockAddress[0];//thread 0 will store the partial thread of the block based on the in place methodology
		//Hence, we store the first element of blockAddress in partial sum of each blockIdx.x 
	}
    //Commit on the CPU
}

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

__global__ void d_L_2(float* in, float* v, float* hold, int k, int size,int nx){
    int idx = threadIdx.x+(blockDim.x*blockIdx.x);
    int tid = threadIdx.x;
    if(idx>k){
        v[idx]=in[idx*nx+k];
    }
    __syncthreads();
    hold[idx]=powf(v[idx],2.0f);
    __syncthreads();
    float* blockAddress = hold + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block

	if (idx >= size)
	{
		return;//Leave the kernel if the global index of our thread exceeds the size of the vector being processed
	}
	//Perform the interleaved reduction, used to reduce divergence.
	//Start adding elements blockDim.x apart, store in place and then half the stride and continue until stride=1
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride && tid + stride < size)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
	if (tid == 0)
	{
		hold[blockIdx.x] = blockAddress[0];//thread 0 will store the partial thread of the block based on the in place methodology
		//Hence, we store the first element of blockAddress in partial sum of each blockIdx.x 
	}
    //Commit on the CPU

}


__host__ void Bidiag_Helper(float* A, float* ref, float* p, float* p_2, float* res_1, float* res_2, float* v_1, float* v_2, int ny, int nx){
    float* d_A;
    float* d_p_row;
    float* d_p_col;
    float* d_res_row;
    float* d_res_col;
    float* d_v_row;
    float* d_v_col;
    float* hold_dot_col;
    float* hold_dot_row;
    float* hold_L_2_col;
    float* hold_L_2_row;
    float L_2{};
    int mat_size=nx*ny*sizeof(float);
    int col_size=ny*sizeof(float);
    int row_size=nx*sizeof(float);
    int block_dim_1D_row=256;
    int grid_dim_1D_row=nx/256+1;
    int block_dim_1D_col=256;
    int grid_dim_1D_col=ny/256+1;
    int block_dim_x_2D=16;
    int block_dim_y_2D=16;
    int grid_dim_x_2D=nx/block_dim_x_2D+1;
    int grid_dim_y_2D=ny/block_dim_y_2D+1;
    HandleCUDAError(cudaMalloc((void**) d_A, mat_size));
    HandleCUDAError(cudaMalloc((void**) d_p_row, row_size));
    HandleCUDAError(cudaMalloc((void**) d_p_col, col_size));
    HandleCUDAError(cudaMalloc((void**) d_res_col, mat_size));
    HandleCUDAError(cudaMalloc((void**) d_res_row, mat_size));
    HandleCUDAError(cudaMalloc((void**) d_v_col, col_size));
    HandleCUDAError(cudaMalloc((void**) d_v_row, row_size));
    HandleCUDAError(cudaMalloc((void**) hold_dot_col, col_size));
    HandleCUDAError(cudaMalloc((void**) hold_dot_row, row_size));
    HandleCUDAError(cudaMalloc((void**) hold_L_2_col, col_size));
    HandleCUDAError(cudaMalloc((void**) hold_L_2_row, row_size));

    HandleCUDAError(cudaMemcpy(d_A,A,mat_size,cudaMemcpyHostToDevice));

    for(int i{}; i<nx;i++){
        d_L_2<<<grid_dim_1D_col,block_dim_1D_col>>>(A,d_v_col,hold_L_2_col,i,ny,nx);
    }





}