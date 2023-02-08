#include "SVD.h"
#include "GPUErrors.h"


__global__ void Outer_Product(float* w, float* v, float* out, int k, int ny, int nx){
    int row=threadIdx.y+(blockDim.y*blockIdx.y);
    int col=threadIdx.x+(blockDim.x*blockIdx.x);
	if(row<ny & col<nx & row>k & col>k){
		out[row*ny+col]=w[row]*v[col];
	}
}

__global__ void Update_v(float* v, int k, float mu, int size){
	float s{};
	float v_start{};
	int tid=threadIdx.x;
	int idx=threadIdx.x+(blockDim.x*blockIdx.x)+k;
	if(idx==k){
		v_start=v[idx];
		if(v[idx]>0){
			s=1.0f;
		}
		else{
			s=-1.0f;
		}
		v[idx]=1.0f;
	}
	if(idx<=size){
		v[idx]/=(v_start+s*mu);
	}
	__syncthreads();

}

__global__ void d_Dot_Product(float* w, float* v, float* hold, int k, int size){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x)+k;
    int tid=threadIdx.x;
	if (idx >= size)
	{
		return;//Leave the kernel if the global index of our thread exceeds the size of the vector being processed
	}
	hold[idx]=w[idx]*v[idx];
    __syncthreads();
    float* blockAddress = hold + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
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

__global__ void Mat_Add_Row(float* Mat_A, float* Mat_B, float* Mat_C, int ny, int nx, int k){
	int row=threadIdx.y+(blockDim.x*blockIdx.x)+k;
	int col
}



__global__ void d_L_2(float* in, float* v, float* hold, int k, int size,int nx){
    int idx = threadIdx.x+(blockDim.x*blockIdx.x)+k;// Push off by k in order to get elements k through n, or k through m
    int tid = threadIdx.x;
    if(idx>=size){
		return;
    }
	//Load the necessary section of A into the v vector, this will be used later
	v[idx]=in[idx*nx+k];
    __syncthreads();
    hold[idx]=powf(v[idx],2.0f);//Update hold rather than v so we can use v in the next step
    __syncthreads();
    float* blockAddress = hold + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
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

}


__global__ void Aug_MatrixVectorMult_Col(float* g_Matrix, float* g_V, float* g_P, int k, const int Size) {
	int row = threadIdx.x + (blockDim.x * blockIdx.x)+k;//We are providing this automatic variable to allow each thread to identify its location
	//Each thread will calculate each entry in our resulting vector
	//To do so, each thread will extract a row of g_Matrix to do with the vector g_V
	float fSum = 0.0f;//We create an automatic variable fSum for each thread to lower memory accesses in the for loop
	//We are going to use fSum instead of writing g_P[row]+=....
	if (row < Size) {
		//We are trying to ensure we are not using more threads than data we have
		for (int j=k+1; j < Size;j++) {
			fSum += g_Matrix[row * Size + j] * g_V[j];//Here we are dotting the row of g_matrix(corresponding to the index of each thread) with g_V
		}
		g_P[row] = fSum;//We now assign the row_th entry of g_P the value fSum, i.e., our dot product
	}
}


__global__ void Aug_MatrixVectorMult_Row(float* g_Matrix, float* g_V, float* g_P, float scalar, int k, const int Size, const int ny) {
	//This is for row.house where w=beta*A^t*v
	int col = threadIdx.x + (blockDim.x * blockIdx.x)+k;//We are providing this automatic variable to allow each thread to identify its location
	//Each thread will calculate each entry in our resulting vector
	//To do so, each thread will extract a row of g_Matrix to do with the vector g_V
	float fSum = 0.0f;//We create an automatic variable fSum for each thread to lower memory accesses in the for loop
	//We are going to use fSum instead of writing g_P[row]+=....
	if (col < Size) {
		//We are trying to ensure we are not using more threads than data we have
		for (int j=k+1; j < ny;j++) {
			fSum += g_Matrix[j * Size + col] * g_V[j];//Here we are dotting the row of g_matrix(corresponding to the index of each thread) with g_V
		}
		g_P[col] = scalar* fSum;//We now assign the row_th entry of g_P the value fSum, i.e., our dot product
	}
}

__global__ void Compute_Beta(float* dot_array, float* beta, int size){
	int tid = threadIdx.x;
	float* blockAddress = dot_array+ (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
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
		*(beta)= -2/blockAddress[0];//thread 0 will store the partial thread of the block based on the in place methodology
		//Hence, we store the first element of blockAddress in partial sum of each blockIdx.x 
	}
}

__global__ void final_L_2(float* sum_arr, int size, float* mu){
	int tid = threadIdx.x;
	float* blockAddress = sum_arr+ (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
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
		*(mu)= blockAddress[0];//thread 0 will store the partial thread of the block based on the in place methodology
		//Hence, we store the first element of blockAddress in partial sum of each blockIdx.x 
	}
}


__host__ void Bidiag_Helper_1(float* A, float* ref, float* p, float* p_2, float* res_1, float* res_2, float* v_1, float* v_2, int ny, int nx){
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
    float mu{};
	float beta{};
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
		final_L_2<<<1,grid_dim_1D_col>>>(hold_L_2_col,grid_dim_1D_col,&mu);
		Update_v<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,i,mu,ny);
		d_Dot_Product<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,d_v_col,hold_dot_col,i,ny);
		Compute_Beta<<<1,grid_dim_1D_col>>>(hold_dot_col,&beta,grid_dim_1D_col);
		Aug_MatrixVectorMult_Row<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,d_p_row,beta,i,nx,ny);



    }





}


__host__ void Bidiag_Helper_2(float* A, float* ref, float* p, float* p_2, float* res_1, float* res_2, float* v_1, float* v_2, int ny, int nx){
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