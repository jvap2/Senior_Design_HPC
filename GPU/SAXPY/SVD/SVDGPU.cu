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
	int idx=threadIdx.x+(blockDim.x*blockIdx.x)+k;
	if(idx<size && idx!=k){
		v_start=v[k];
		if(fabsf(v_start)>1e-6){
			s=v_start/fabsf(v_start);
		}
		float beta=v_start+s*(*mu);
		if(fabsf(beta)>1e-6){
			v[idx]/=beta;
		}
	}
	if(idx==k){
		v[idx]=1.0f;
	}
	__syncthreads();

}

__global__ void d_Dot_Product(float* v, float* hold, float* d_psum, int k, int size){
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
		d_psum[blockIdx.x] = blockAddress[0];//thread 0 will store the partial thread of the block based on the in place methodology
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

__global__ void Compute_Beta(float* dot_array, float* beta){
	int tid = threadIdx.x;
	// float* blockAddress = dot_array + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block
	float* blockAddress = dot_array;
	//Perform the interleaved reduction, used to reduce divergence.
	//Start adding elements blockDim.x apart, store in place and then half the stride and continue until stride=1
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
	if (tid == 0 && blockAddress[tid]!=0)
	{
		*(beta)= -2/blockAddress[0];//thread 0 will store the partial thread of the block based on the in place methodology
		//Hence, we store the first element of blockAddress in partial sum of each blockIdx.x 
	}
}


__global__ void d_L_2_Partial_Reduction(float* in, float* d_v,float* hold_vect,float* g_PartialSum, int k,int size,int nx){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x)+k;
    int tid=threadIdx.x;
    if (idx>=size){
        return;
    }
	d_v[idx]=in[idx*nx+k];
	__syncthreads();
    hold_vect[idx]=d_v[idx]*d_v[idx];
    __syncthreads();
    float* blockAddress=hold_vect+(blockDim.x*blockIdx.x);
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
    if(tid==0){
        g_PartialSum[blockIdx.x]=blockAddress[0];
    }
}

__global__ void d_L_2_Partial_Reduction_CH(float* in, float* d_v, float* hold_vect,float* g_PartialSum, int k, int size,int nx){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x)+k+1;
    int tid=threadIdx.x;
    if (idx>=size){
        return;
    }
	d_v[idx]=in[k*nx+idx];
    __syncthreads();
    hold_vect[idx]=d_v[idx]*d_v[idx];
    __syncthreads();
    float* blockAddress=hold_vect+(blockDim.x*blockIdx.x);
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
    if(tid==0){
        g_PartialSum[blockIdx.x]=blockAddress[0];
    }
}

__global__ void d_Commit_L_2(float* g_Partial_Sum, float* mu){
    int tid=threadIdx.x;
    float* blockAddress=g_Partial_Sum;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
    if(tid==0){
        *(mu)=sqrtf(blockAddress[0]);
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
    float* mu_col{};
	float* beta_col{};
	float* mu_row{};
	float* beta_row{};
	float* d_P_sum_L_2_col;
	float* d_P_sum_L_2_row;
	float* d_P_sum_dot_col;
	float* d_P_sum_dot_row;
	float temp_vec[ny];
	float temp[1];
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
	dim3 block(block_dim_x_2D,block_dim_y_2D);
	dim3 grid(grid_dim_x_2D,grid_dim_y_2D);
	int col_psum_size=sizeof(float)*grid_dim_1D_col;
	int row_psum_size=sizeof(float)*grid_dim_1D_row;
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
    HandleCUDAError(cudaMalloc((void**) &d_P_sum_dot_col, col_psum_size));
    HandleCUDAError(cudaMalloc((void**) &d_P_sum_dot_row, row_psum_size));
    HandleCUDAError(cudaMalloc((void**) &d_P_sum_L_2_col, col_psum_size));
    HandleCUDAError(cudaMalloc((void**) &d_P_sum_L_2_row, row_psum_size));

    HandleCUDAError(cudaMalloc((void**) &mu_col, sizeof(float)));
    HandleCUDAError(cudaMalloc((void**) &beta_col, sizeof(float)));
	HandleCUDAError(cudaMalloc((void**) &mu_row, sizeof(float)));
    HandleCUDAError(cudaMalloc((void**) &beta_row, sizeof(float)));
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
		HandleCUDAError(cudaMemset(mu_col,0,sizeof(float)));
		HandleCUDAError(cudaMemset(mu_row,0,sizeof(float)));
		HandleCUDAError(cudaMemset(beta_col,0,sizeof(float)));
		HandleCUDAError(cudaMemset(beta_row,0,sizeof(float)));
		int j=i+1;
        d_L_2_Partial_Reduction<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,hold_L_2_col,d_P_sum_L_2_col,i,ny,nx);
		cudaDeviceSynchronize();
		d_Commit_L_2<<<1,grid_dim_1D_col>>>(d_P_sum_L_2_col,mu_col);
		cudaDeviceSynchronize();
		Update_v<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,i,mu_col,ny);
		cudaDeviceSynchronize();
		d_Dot_Product<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,hold_dot_col,d_P_sum_dot_col,i,ny);
		cudaDeviceSynchronize();
		Compute_Beta<<<1,grid_dim_1D_col>>>(d_P_sum_dot_col,beta_col);
		cudaDeviceSynchronize();
		Aug_MatrixVectorMult_Row<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,d_p_row,beta_col,i,nx,ny);
		cudaDeviceSynchronize();
		d_Outer_Product<<<grid,block>>>(d_p_row,d_v_col,d_res_col,i,ny,nx);
		cudaDeviceSynchronize();
		Mat_Add_Row<<<grid,block>>>(d_A,d_res_col,d_A,ny,nx,i);
		cudaDeviceSynchronize();
		d_L_2_Partial_Reduction_CH<<<grid_dim_1D_row,block_dim_1D_row>>>(d_A,d_v_row,hold_L_2_row,d_P_sum_L_2_row,i,nx,nx);
		cudaDeviceSynchronize();
		d_Commit_L_2<<<1,grid_dim_1D_row>>>(d_P_sum_L_2_row,mu_row);
		cudaDeviceSynchronize();
		Update_v<<<grid_dim_1D_row,block_dim_1D_row>>>(d_v_row,j,mu_row,nx);
		cudaDeviceSynchronize();
		d_Dot_Product<<<grid_dim_1D_row,block_dim_1D_row>>>(d_v_row,hold_dot_row,d_P_sum_dot_row,j,nx);
		cudaDeviceSynchronize();
		Compute_Beta<<<1,grid_dim_1D_row>>>(d_P_sum_dot_row,beta_row);
		cudaDeviceSynchronize();
		Aug_MatrixVectorMult_Col<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_row,d_p_col,beta_row,i,nx,ny); //should this be d_p_col or d_p_row?
		cudaDeviceSynchronize();
		d_Outer_Product<<<grid,block>>>(d_v_row,d_p_col,d_res_row,i,ny,nx);
		cudaDeviceSynchronize();
		Mat_Add_Col<<<grid,block>>>(d_A,d_res_row,d_A,ny,nx,i);
		cudaDeviceSynchronize();
    }
	if(!HandleCUDAError(cudaMemcpy(temp_vec,d_v_row,col_size,cudaMemcpyDeviceToHost))){
		cout<<"cannot display"<<endl;
	}
	if(!HandleCUDAError(cudaMemcpy(temp,beta_col,sizeof(float),cudaMemcpyDeviceToHost))){
		cout<<"cannot display"<<endl;
	}
	cout<<"mu="<<(temp[0])<<endl;
	DisplayMatrix("dot",temp_vec,ny,1);
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
	HandleCUDAError(cudaFree(mu_col));
	HandleCUDAError(cudaFree(mu_row));
	HandleCUDAError(cudaFree(beta_col));
	HandleCUDAError(cudaFree(beta_row));

	cudaDeviceReset();
}


__host__ void Bidiag_Helper_1(float* A, float* ref,  int ny, int nx){
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
	float* d_P_sum_L_2_col;
	float* d_P_sum_L_2_row;
	float* d_P_sum_dot_col;
	float* d_P_sum_dot_row;
    float* mu_col{};
	float* beta_col{};
	float* mu_row{};
	float* beta_row{};
	float temp_vec[ny];
	float temp;
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
	dim3 block(block_dim_x_2D,block_dim_y_2D);
	dim3 grid(grid_dim_x_2D,grid_dim_y_2D);
	int col_psum_size=sizeof(float)*grid_dim_1D_col;
	int row_psum_size=sizeof(float)*grid_dim_1D_row;
	cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;
	cout << "\t1D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along col dimension: " << grid_dim_1D_col << endl;
	cout << "\tNumber of Blocks along row dimension: " << grid_dim_1D_row << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along col dimension: " << block_dim_1D_col << endl;
	cout << "\tNumber of threads along row dimension: " << block_dim_1D_row << endl;
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
	HandleCUDAError(cudaMalloc((void**) &d_P_sum_dot_col, col_psum_size));
    HandleCUDAError(cudaMalloc((void**) &d_P_sum_dot_row, row_psum_size));
    HandleCUDAError(cudaMalloc((void**) &d_P_sum_L_2_col, col_psum_size));
    HandleCUDAError(cudaMalloc((void**) &d_P_sum_L_2_row, row_psum_size));
    HandleCUDAError(cudaMalloc((void**) &mu_col, sizeof(float)));
    HandleCUDAError(cudaMalloc((void**) &beta_col, sizeof(float)));
	HandleCUDAError(cudaMalloc((void**) &mu_row, sizeof(float)));
    HandleCUDAError(cudaMalloc((void**) &beta_row, sizeof(float)));

    HandleCUDAError(cudaMemcpy(d_A,A,mat_size,cudaMemcpyHostToDevice));

    for(int i{}; i<(nx);i++){
		HandleCUDAError(cudaMemset(d_p_col,0,col_size));
		HandleCUDAError(cudaMemset(d_res_col,0,mat_size));
		HandleCUDAError(cudaMemset(d_v_col,0,col_size));
		HandleCUDAError(cudaMemset(hold_dot_col,0,col_size));
		HandleCUDAError(cudaMemset(hold_L_2_col,0,col_size));
		HandleCUDAError(cudaMemset(mu_col,0,sizeof(float)));
		HandleCUDAError(cudaMemset(beta_col,0,sizeof(float)));
		HandleCUDAError(cudaMemset(d_P_sum_dot_col,0,col_psum_size));
		HandleCUDAError(cudaMemset(d_P_sum_L_2_col,0,col_psum_size));

		int j=i+1;
        d_L_2_Partial_Reduction<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,hold_L_2_col,d_P_sum_L_2_col,i,ny,nx);
		cudaDeviceSynchronize();
		d_Commit_L_2<<<1,grid_dim_1D_col>>>(d_P_sum_L_2_col,mu_col);
		cudaDeviceSynchronize();
		Update_v<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,i,mu_col,ny);
		cudaDeviceSynchronize();
		d_Dot_Product<<<grid_dim_1D_col,block_dim_1D_col>>>(d_v_col,hold_dot_col,d_P_sum_dot_col,i,ny);
		cudaDeviceSynchronize();
		Compute_Beta<<<1,grid_dim_1D_col>>>(d_P_sum_dot_col,beta_col);
		cudaDeviceSynchronize();
		Aug_MatrixVectorMult_Row<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_col,d_p_row,beta_col,i,nx,ny);
		cudaDeviceSynchronize();
		d_Outer_Product<<<grid,block>>>(d_p_row,d_v_col,d_res_col,i,ny,nx);
		cudaDeviceSynchronize();
		Mat_Add_Row<<<grid,block>>>(d_A,d_res_col,d_A,ny,nx,i);
		cudaDeviceSynchronize();
		if(i<(nx-2)){
			HandleCUDAError(cudaMemset(d_p_row,0,row_size));
			HandleCUDAError(cudaMemset(d_res_row,0,mat_size));
			HandleCUDAError(cudaMemset(d_v_row,0,row_size));
			HandleCUDAError(cudaMemset(hold_dot_row,0,row_size));
			HandleCUDAError(cudaMemset(hold_L_2_row,0,row_size));
			HandleCUDAError(cudaMemset(mu_row,0,sizeof(float)));
			HandleCUDAError(cudaMemset(beta_row,0,sizeof(float)));
			HandleCUDAError(cudaMemset(d_P_sum_dot_row,0,row_psum_size));
			HandleCUDAError(cudaMemset(d_P_sum_L_2_row,0,row_psum_size));
			d_L_2_Partial_Reduction_CH<<<grid_dim_1D_row,block_dim_1D_row>>>(d_A,d_v_row,hold_dot_row,d_P_sum_L_2_row,i,nx,nx);
			cudaDeviceSynchronize();
			d_Commit_L_2<<<1,grid_dim_1D_row>>>(d_P_sum_L_2_row,mu_row);
			cudaDeviceSynchronize();
			Update_v<<<grid_dim_1D_row,block_dim_1D_row>>>(d_v_row,j,mu_row,nx);
			cudaDeviceSynchronize();
			d_Dot_Product<<<grid_dim_1D_row,block_dim_1D_row>>>(d_v_row,hold_dot_row,d_P_sum_dot_row,j,nx);
			cudaDeviceSynchronize();
			Compute_Beta<<<1,grid_dim_1D_row>>>(d_P_sum_dot_row,beta_row);
			cudaDeviceSynchronize();
			Aug_MatrixVectorMult_Col<<<grid_dim_1D_col,block_dim_1D_col>>>(d_A,d_v_row,d_p_col,beta_row,i,nx,ny); //should this be d_p_col or d_p_row?
			cudaDeviceSynchronize();
			d_Outer_Product<<<grid,block>>>(d_v_row,d_p_col,d_res_row,i,ny,nx);
			cudaDeviceSynchronize();
			Mat_Add_Col<<<grid,block>>>(d_A,d_res_row,d_A,ny,nx,i);
			cudaDeviceSynchronize();
		}
    }
	if(!HandleCUDAError(cudaMemcpy(temp_vec,hold_L_2_col,col_size,cudaMemcpyDeviceToHost))){
		cout<<"cannot display"<<endl;
	}
	if(!HandleCUDAError(cudaMemcpy(&temp,mu_col,sizeof(float),cudaMemcpyDeviceToHost))){
		cout<<"cannot display"<<endl;
	}
	cout<<"mu="<<(temp)<<endl;

	for(int j{}; j<ny; j++){
		cout<<temp_vec[j]<<endl;
	}

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
	HandleCUDAError(cudaFree(mu_col));
	HandleCUDAError(cudaFree(mu_row));
	HandleCUDAError(cudaFree(beta_col));
	HandleCUDAError(cudaFree(beta_row));

	cudaDeviceReset();

}