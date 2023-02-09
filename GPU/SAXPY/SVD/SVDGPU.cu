#include "SVD.h"
#include "GPUErrors.h"


__global__ void d_Outer_Product(float* w, float* v, float* out, int k, int ny, int nx){
    int row=threadIdx.y+(blockDim.y*blockIdx.y)+k;
    int col=threadIdx.x+(blockDim.x*blockIdx.x)+k;
	if(row<ny && col<nx){
		out[row*nx+col]=w[col]*v[row];
	}
}

__global__ void Update_v(float* v, int k, float* mu, int size){
	float s{};
	float v_start{};
	int tid=threadIdx.x;
	int idx=threadIdx.x+(blockDim.x*blockIdx.x)+k;
	if(idx<size && idx!=k){
		v_start=v[k];
		if(v_start>0){
			s=1.0f;
		}
		else if(v_start<0){
			s=-1.0f;
		}
		else{
			s=0.0;
		}
		float beta=v_start+s*(*mu);
		if(fabsf(beta)>1e-6){
			v[idx]/=(v_start+s*(*mu));
		}
	}
	if(idx==k){
		v[idx]=1.0f;
	}
	__syncthreads();

}

__global__ void d_Dot_Product(float* v, float* hold, int k, int size){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x)+k;
    int tid=threadIdx.x;
	hold[idx]=powf(v[idx],2.0f);
    __syncthreads();
    float* blockAddress = hold + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
	//Perform the interleaved reduction, used to reduce divergence.
	//Start adding elements blockDim.x apart, store in place and then half the stride and continue until stride=1
	if (idx >= size)
	{
		return;//Leave the kernel if the global index of our thread exceeds the size of the vector being processed
	}
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
	int row=threadIdx.y+(blockDim.y*blockIdx.y)+k;
	int col=threadIdx.x+(blockDim.x*blockIdx.x)+k;
	if(row<ny & col<nx){
		Mat_C[row*nx+col]=Mat_A[row*nx+col]+Mat_B[row*nx+col];
	}
}

__global__ void Mat_Add_Col(float* Mat_A, float* Mat_B, float* Mat_C, int ny, int nx, int k){
	int row=threadIdx.y+(blockDim.y*blockIdx.y)+k;
	int col=threadIdx.x+(blockDim.x*blockIdx.x)+k+1;
	if(row<ny & col<nx){
		Mat_C[row*nx+col]=Mat_A[row*nx+col]+Mat_B[row*nx+col];
	}
}



__global__ void d_L_2(float* in, float* v, float* hold, int k, int size,int nx){
    int idx = threadIdx.x+(blockDim.x*blockIdx.x)+k;// Push off by k in order to get elements k through n, or k through m
    int tid = threadIdx.x;
	//Load the necessary section of A into the v vector, this will be used later
	v[idx]=in[idx*nx+k];
    __syncthreads();
    hold[idx]=powf(v[idx],2.0f);//Update hold rather than v so we can use v in the next step
    __syncthreads();
    float* blockAddress = hold + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
	//Perform the interleaved reduction, used to reduce divergence.
	//Start adding elements blockDim.x apart, store in place and then half the stride and continue until stride=1
	if(idx>=size){
		return;
    }
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

__global__ void d_L_2_CH(float* in, float* v, float* hold, int k, int size,int nx){
    int idx = threadIdx.x+(blockDim.x*blockIdx.x)+k+1;// Push off by k in order to get elements k through n, or k through m
    int tid = threadIdx.x;
	//Load the necessary section of A into the v vector, this will be used later
	v[idx]=in[k*nx+idx];
    __syncthreads();
    hold[idx]=powf(v[idx],2.0f);//Update hold rather than v so we can use v in the next step
    __syncthreads();
    float* blockAddress = hold + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
	//Perform the interleaved reduction, used to reduce divergence.
	//Start adding elements blockDim.x apart, store in place and then half the stride and continue until stride=1
	if(idx>=size){
		return;
    }
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


__global__ void Aug_MatrixVectorMult_Col(float* g_Matrix, float* g_V, float* g_P, float* scalar,int k, const int nx, const int ny) {
	int row = threadIdx.x + (blockDim.x * blockIdx.x)+k;//We are providing this automatic variable to allow each thread to identify its location
	//Each thread will calculate each entry in our resulting vector
	//To do so, each thread will extract a row of g_Matrix to do with the vector g_V
	float fSum = 0.0f;//We create an automatic variable fSum for each thread to lower memory accesses in the for loop
	//We are going to use fSum instead of writing g_P[row]+=....
	if (row < ny) {
		//We are trying to ensure we are not using more threads than data we have
		for (int j=k+1; j < nx;j++) {
			fSum += g_Matrix[row * nx + j] * g_V[j];//Here we are dotting the row of g_matrix(corresponding to the index of each thread) with g_V
		}
		g_P[row] = (*scalar)*fSum;//We now assign the row_th entry of g_P the value fSum, i.e., our dot product
	}
}


__global__ void Aug_MatrixVectorMult_Row(float* g_Matrix, float* g_V, float* g_P, float* scalar, int k, const int nx, const int ny) {
	//This is for row.house where w=beta*A^t*v
	int col = threadIdx.x + (blockDim.x * blockIdx.x)+k;//We are providing this automatic variable to allow each thread to identify its location
	//Each thread will calculate each entry in our resulting vector
	//To do so, each thread will extract a row of g_Matrix to do with the vector g_V
	float fSum = 0.0f;//We create an automatic variable fSum for each thread to lower memory accesses in the for loop
	//We are going to use fSum instead of writing g_P[row]+=....
	if (col < nx) {
		//We are trying to ensure we are not using more threads than data we have
		for (int j=k; j < ny;j++) {
			fSum += g_Matrix[j * nx + col] * g_V[j];//Here we are dotting the row of g_matrix(corresponding to the index of each thread) with g_V
		}
		g_P[col] =(*scalar)*fSum;//We now assign the row_th entry of g_P the value fSum, i.e., our dot product
	}
}

__global__ void Compute_Beta(float* dot_array, float* beta, int size){
	int tid = threadIdx.x;
	float* blockAddress = dot_array + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
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
		*(mu)= sqrt(blockAddress[0]);//thread 0 will store the partial thread of the block based on the in place methodology
		//Hence, we store the first element of blockAddress in partial sum of each blockIdx.x 
	}
}


__host__ void Bidiag_Helper_2(float* A, float* ref,  int ny, int nx){
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
    float* mu{};
	float* beta{};
	float temp[ny];
	// float temp{};
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
	dim3 block(block_dim_y_2D,block_dim_x_2D);
	dim3 grid(grid_dim_y_2D,grid_dim_x_2D);
    HandleCUDAError(cudaMalloc((void**) &d_A, mat_size));
    HandleCUDAError(cudaMalloc((void**) &d_p_row, row_size));
    HandleCUDAError(cudaMalloc((void**) &d_p_col, col_size));
    HandleCUDAError(cudaMalloc((void**) &d_res_col, mat_size));
    HandleCUDAError(cudaMalloc((void**) &d_res_row, mat_size));
    HandleCUDAError(cudaMalloc((void**) &d_v_col, col_size));
    HandleCUDAError(cudaMalloc((void**) &d_v_row, row_size));
    HandleCUDAError(cudaMalloc((void**) &hold_dot_col, col_size));
    HandleCUDAError(cudaMalloc((void**) &hold_dot_row, row_size));
    HandleCUDAError(cudaMalloc((void**) &hold_L_2_col, col_size));
    HandleCUDAError(cudaMalloc((void**) &hold_L_2_row, row_size));
    HandleCUDAError(cudaMalloc((void**) &mu, sizeof(float)));
    HandleCUDAError(cudaMalloc((void**) &beta, sizeof(float)));
	HandleCUDAError(cudaMemset(d_p_row,0,row_size));
	HandleCUDAError(cudaMemset(d_p_col,0,col_size));
	HandleCUDAError(cudaMemset(d_res_col,0,mat_size));
	HandleCUDAError(cudaMemset(d_res_row,0,mat_size));
	HandleCUDAError(cudaMemset(d_v_col,0,col_size));
	HandleCUDAError(cudaMemset(d_v_row,0,row_size));
	HandleCUDAError(cudaMemset(hold_dot_col,0,col_size));
	HandleCUDAError(cudaMemset(hold_dot_row,0,row_size));
	HandleCUDAError(cudaMemset(hold_L_2_col,0,col_size));
	HandleCUDAError(cudaMemset(hold_L_2_row,0,row_size));
    HandleCUDAError(cudaMemcpy(d_A,A,mat_size,cudaMemcpyHostToDevice));

    for(int i{}; i<(nx-2);i++){
		HandleCUDAError(cudaMemset(d_p_row,0,row_size));
		HandleCUDAError(cudaMemset(d_p_col,0,col_size));
		HandleCUDAError(cudaMemset(d_res_col,0,mat_size));
		HandleCUDAError(cudaMemset(d_res_row,0,mat_size));
		HandleCUDAError(cudaMemset(d_v_col,0,col_size));
		HandleCUDAError(cudaMemset(d_v_row,0,row_size));
		HandleCUDAError(cudaMemset(hold_dot_col,0,col_size));
		HandleCUDAError(cudaMemset(hold_dot_row,0,row_size));
		HandleCUDAError(cudaMemset(hold_L_2_col,0,col_size));
		HandleCUDAError(cudaMemset(hold_L_2_row,0,row_size));
		int j=i+1;
        d_L_2<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,hold_L_2_col,i,ny,nx);
		cudaDeviceSynchronize();
		final_L_2<<<1,grid_dim_1D_col>>>(hold_L_2_col,grid_dim_1D_col,mu);
		cudaDeviceSynchronize();
		Update_v<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,i,mu,ny);
		cudaDeviceSynchronize();
		d_Dot_Product<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,hold_dot_col,i,ny);
		cudaDeviceSynchronize();
		Compute_Beta<<<1,grid_dim_1D_col>>>(hold_dot_col,beta,grid_dim_1D_col);
		cudaDeviceSynchronize();
		Aug_MatrixVectorMult_Row<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,d_p_row,beta,i,nx,ny);
		cudaDeviceSynchronize();
		d_Outer_Product<<<grid,block>>>(d_p_row,d_v_col,d_res_col,i,ny,nx);
		cudaDeviceSynchronize();
		Mat_Add_Row<<<grid,block>>>(d_A,d_res_col,d_A,ny,nx,i);
		cudaDeviceSynchronize();
		d_L_2_CH<<<grid_dim_1D_row,block_dim_1D_row>>>(d_A,d_v_row,hold_L_2_row,i,nx,nx);
		cudaDeviceSynchronize();
		final_L_2<<<1,grid_dim_1D_row>>>(hold_L_2_row,grid_dim_1D_row,mu);
		cudaDeviceSynchronize();
		Update_v<<<grid_dim_1D_row,block_dim_1D_row>>>(d_v_row,j,mu,nx);
		cudaDeviceSynchronize();
		d_Dot_Product<<<grid_dim_1D_row,block_dim_1D_row>>>(d_v_row,hold_dot_row,j,nx);
		cudaDeviceSynchronize();
		Compute_Beta<<<1,grid_dim_1D_row>>>(hold_dot_row,beta,grid_dim_1D_row);
		cudaDeviceSynchronize();
		Aug_MatrixVectorMult_Col<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_row,d_p_row,beta,i,nx,ny);
		cudaDeviceSynchronize();
		d_Outer_Product<<<grid,block>>>(d_v_row,d_p_row,d_res_col,i,ny,nx);
		cudaDeviceSynchronize();
		Mat_Add_Col<<<grid,block>>>(d_A,d_res_col,d_A,ny,nx,i);
		cudaDeviceSynchronize();
    }
	// if(!HandleCUDAError(cudaMemcpy(temp,d_p_row,col_size,cudaMemcpyDeviceToHost))){
	// 	cout<<"cannot display"<<endl;
	// }
	// // cout<<"mu="<<(temp)<<endl;
	// DisplayMatrix("w",temp,ny,1);
	HandleCUDAError(cudaMemcpy(A,d_A,mat_size,cudaMemcpyDeviceToHost));
	SVDVerification(ref,A,ny,nx);
	DisplayMatrix("GPU A",A,ny,nx);
	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_p_row));
	HandleCUDAError(cudaFree(d_p_col));
	HandleCUDAError(cudaFree(d_res_col));
	HandleCUDAError(cudaFree(d_res_row));
	HandleCUDAError(cudaFree(d_v_col));
	HandleCUDAError(cudaFree(d_v_row));
	HandleCUDAError(cudaFree(hold_dot_col));
	HandleCUDAError(cudaFree(hold_dot_row));
	HandleCUDAError(cudaFree(hold_L_2_col));
	HandleCUDAError(cudaFree(hold_L_2_row));
	cudaDeviceReset();
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
    float* mu{};
	float* beta{};
	float temp[ny];
	// float temp{};
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
	dim3 block(block_dim_y_2D,block_dim_x_2D);
	dim3 grid(grid_dim_y_2D,grid_dim_x_2D);
    HandleCUDAError(cudaMalloc((void**) &d_A, mat_size));
    HandleCUDAError(cudaMalloc((void**) &d_p_row, row_size));
    HandleCUDAError(cudaMalloc((void**) &d_p_col, col_size));
    HandleCUDAError(cudaMalloc((void**) &d_res_col, mat_size));
    HandleCUDAError(cudaMalloc((void**) &d_res_row, mat_size));
    HandleCUDAError(cudaMalloc((void**) &d_v_col, col_size));
    HandleCUDAError(cudaMalloc((void**) &d_v_row, row_size));
    HandleCUDAError(cudaMalloc((void**) &hold_dot_col, col_size));
    HandleCUDAError(cudaMalloc((void**) &hold_dot_row, row_size));
    HandleCUDAError(cudaMalloc((void**) &hold_L_2_col, col_size));
    HandleCUDAError(cudaMalloc((void**) &hold_L_2_row, row_size));
    HandleCUDAError(cudaMalloc((void**) &mu, sizeof(float)));
    HandleCUDAError(cudaMalloc((void**) &beta, sizeof(float)));
	HandleCUDAError(cudaMemset(d_p_row,0,row_size));
	HandleCUDAError(cudaMemset(d_p_col,0,col_size));
	HandleCUDAError(cudaMemset(d_res_col,0,mat_size));
	HandleCUDAError(cudaMemset(d_res_row,0,mat_size));
	HandleCUDAError(cudaMemset(d_v_col,0,col_size));
	HandleCUDAError(cudaMemset(d_v_row,0,row_size));
	HandleCUDAError(cudaMemset(hold_dot_col,0,col_size));
	HandleCUDAError(cudaMemset(hold_dot_row,0,row_size));
	HandleCUDAError(cudaMemset(hold_L_2_col,0,col_size));
	HandleCUDAError(cudaMemset(hold_L_2_row,0,row_size));
    HandleCUDAError(cudaMemcpy(d_A,A,mat_size,cudaMemcpyHostToDevice));

    for(int i{}; i<(nx);i++){
		HandleCUDAError(cudaMemset(d_p_row,0,row_size));
		HandleCUDAError(cudaMemset(d_p_col,0,col_size));
		HandleCUDAError(cudaMemset(d_res_col,0,mat_size));
		HandleCUDAError(cudaMemset(d_res_row,0,mat_size));
		HandleCUDAError(cudaMemset(d_v_col,0,col_size));
		HandleCUDAError(cudaMemset(d_v_row,0,row_size));
		HandleCUDAError(cudaMemset(hold_dot_col,0,col_size));
		HandleCUDAError(cudaMemset(hold_dot_row,0,row_size));
		HandleCUDAError(cudaMemset(hold_L_2_col,0,col_size));
		HandleCUDAError(cudaMemset(hold_L_2_row,0,row_size));
		int j=i+1;
        d_L_2<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,hold_L_2_col,i,ny,nx);
		cudaDeviceSynchronize();
		final_L_2<<<1,grid_dim_1D_col>>>(hold_L_2_col,grid_dim_1D_col,mu);
		cudaDeviceSynchronize();
		Update_v<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,i,mu,ny);
		cudaDeviceSynchronize();
		d_Dot_Product<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,hold_dot_col,i,ny);
		cudaDeviceSynchronize();
		Compute_Beta<<<1,grid_dim_1D_col>>>(hold_dot_col,beta,grid_dim_1D_col);
		cudaDeviceSynchronize();
		Aug_MatrixVectorMult_Row<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,d_p_row,beta,i,nx,ny);
		cudaDeviceSynchronize();
		d_Outer_Product<<<grid,block>>>(d_p_row,d_v_col,d_res_col,i,ny,nx);
		cudaDeviceSynchronize();
		Mat_Add_Row<<<grid,block>>>(d_A,d_res_col,d_A,ny,nx,i);
		cudaDeviceSynchronize();
		if(i<=(nx-2)){
			d_L_2_CH<<<grid_dim_1D_row,block_dim_1D_row>>>(d_A,d_v_row,hold_L_2_row,i,nx,nx);
			cudaDeviceSynchronize();
			final_L_2<<<1,grid_dim_1D_row>>>(hold_L_2_row,grid_dim_1D_row,mu);
			cudaDeviceSynchronize();
			Update_v<<<grid_dim_1D_row,block_dim_1D_row>>>(d_v_row,j,mu,nx);
			cudaDeviceSynchronize();
			d_Dot_Product<<<grid_dim_1D_row,block_dim_1D_row>>>(d_v_row,hold_dot_row,j,nx);
			cudaDeviceSynchronize();
			Compute_Beta<<<1,grid_dim_1D_row>>>(hold_dot_row,beta,grid_dim_1D_row);
			cudaDeviceSynchronize();
			Aug_MatrixVectorMult_Col<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_row,d_p_row,beta,i,nx,ny);
			cudaDeviceSynchronize();
			d_Outer_Product<<<grid,block>>>(d_v_row,d_p_row,d_res_col,i,ny,nx);
			cudaDeviceSynchronize();
			Mat_Add_Col<<<grid,block>>>(d_A,d_res_col,d_A,ny,nx,i);
			cudaDeviceSynchronize();
		}
    }
	// if(!HandleCUDAError(cudaMemcpy(temp,d_p_row,col_size,cudaMemcpyDeviceToHost))){
	// 	cout<<"cannot display"<<endl;
	// }
	// // cout<<"mu="<<(temp)<<endl;
	// DisplayMatrix("w",temp,ny,1);
	HandleCUDAError(cudaMemcpy(A,d_A,mat_size,cudaMemcpyDeviceToHost));
	SVDVerification(ref,A,ny,nx);
	DisplayMatrix("GPU A",A,ny,nx);
	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_p_row));
	HandleCUDAError(cudaFree(d_p_col));
	HandleCUDAError(cudaFree(d_res_col));
	HandleCUDAError(cudaFree(d_res_row));
	HandleCUDAError(cudaFree(d_v_col));
	HandleCUDAError(cudaFree(d_v_row));
	HandleCUDAError(cudaFree(hold_dot_col));
	HandleCUDAError(cudaFree(hold_dot_row));
	HandleCUDAError(cudaFree(hold_L_2_col));
	HandleCUDAError(cudaFree(hold_L_2_row));
	cudaDeviceReset();

}