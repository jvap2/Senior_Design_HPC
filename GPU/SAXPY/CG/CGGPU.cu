#include "GPUErrors.h"
#include "CG.h"

__global__ void d_Dot_Partial(float* in_1, float* in_2, float* hold, float* g_PartialSums, int size){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x);
    int tid=threadIdx.x;
    hold[idx]=in_1[idx]*in_2[idx];
    __syncthreads();
    if (idx>=size){
        return;
    }
    float* blockAddress= hold+(blockDim.x*blockIdx.x);
    for(int stride=blockDim.x/2; stride>0; stride>>=1){
        if (tid<stride && tid+stride<size){
            blockAddress[tid]+=blockAddress[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        g_PartialSums[blockIdx.x]=blockAddress[0];
    }
}

__global__ void d_Commit_Dot(float* g_Partial_Sum, float* dot, int* flag){
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
        *(dot)=(blockAddress[0]);
        if(fabsf(*dot)<1e-8){
            *flag=0;
        }
    }
}

__global__ void VectAdd(float* g_A, float* g_B, float* g_C, int size){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x);
    if(idx<size){
        g_C[idx]=g_A[idx]+g_B[idx];
    }
}

__global__ void Const_Vect_Mult(float* vect, float* out, float* scalar, int size){
    int idx =threadIdx.x+(blockDim.x*blockIdx.x);
    if(idx<size){
        out[idx]=*scalar*vect[idx];
    }
}


__global__ void MatrixVectorMult(float* g_Matrix, float* g_V, float* g_P, const int Size) {
	int row = threadIdx.x + (blockDim.x * blockIdx.x);//We are providing this automatic variable to allow each thread to identify its location
	//Each thread will calculate each entry in our resulting vector
	//To do so, each thread will extract a row of g_Matrix to do with the vector g_V
	float fSum = 0.0f;//We create an automatic variable fSum for each thread to lower memory accesses in the for loop
	//We are going to use fSum instead of writing g_P[row]+=....
	if (row < Size) {
		//We are trying to ensure we are not using more threads than data we have
		for (int k{}; k < Size;k++) {
			fSum += g_Matrix[row * Size + k] * g_V[k];//Here we are dotting the row of g_matrix(corresponding to the index of each thread) with g_V
		}
		g_P[row] = fSum;//We now assign the row_th entry of g_P the value fSum, i.e., our dot product
	}
}

__global__ void comp_lamba(float* in, float* in_2, float* out,int size, int flag){
    int idx =threadIdx.x+(blockDim.x*blockIdx.x);
    if(idx<size){
        if(flag==1){
            out[idx]=in[idx]*in_2[idx];
        }
        else{
            out[idx]=-in[idx]*in_2[idx];
        }
    }

}


__host__ void CG_Helper(float* A, float* ref, float* r, float* r_old, float* d, float* d_old, float* x, float* x_old, float beta, float lamdba, int size){
    float* d_Ad;
    float* lamd_d;
    float* beta_d;
    float* lambd_AD;
    float* d_A;
    float* d_r;
    float* d_r_old;
    float* d_d;
    float* d_d_old;
    float* d_x;
    float* d_x_old;
    float* d_beta;
    float* d_lambda;
    float* d_neg_lambda;
    float* temp_1;
    float* temp_2;
    float* d_dot_partial_1;
    float* d_dot_partial_2;
    float* d_hold_1;
    float* d_hold_2;
    int* flag;
    int* flag_2;
    int threads_per_block=128;
    int blocks_per_grid=size/threads_per_block+1;
    int mat_size=size*size*sizeof(float);
    int vect_size=size*sizeof(float);
    int var_size=sizeof(float);
    int flag_size=sizeof(int);
    int p_sum_size=sizeof(float)*blocks_per_grid;
    int host_flag=1;
    
    HandleCUDAError(cudaMalloc((void**) &d_A,mat_size));
    HandleCUDAError(cudaMalloc((void**) &d_r,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_r_old,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_d,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_d_old,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_x,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_x_old,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_beta,var_size));
    HandleCUDAError(cudaMalloc((void**) &d_lambda,var_size));
    HandleCUDAError(cudaMalloc((void**) &d_neg_lambda,var_size));
    HandleCUDAError(cudaMalloc((void**) &temp_1,var_size));
    HandleCUDAError(cudaMalloc((void**) &temp_2,var_size));
    HandleCUDAError(cudaMalloc((void**) &flag,flag_size));
    HandleCUDAError(cudaMalloc((void**) &flag_2,flag_size));
    HandleCUDAError(cudaMalloc((void**) &d_Ad,vect_size));
    HandleCUDAError(cudaMalloc((void**) &lamd_d,vect_size));
    HandleCUDAError(cudaMalloc((void**) &beta_d,mat_size));
    HandleCUDAError(cudaMalloc((void**) &lambd_AD,mat_size));
    HandleCUDAError(cudaMalloc((void**) &d_dot_partial_1,p_sum_size));
    HandleCUDAError(cudaMalloc((void**) &d_dot_partial_2,p_sum_size));
    HandleCUDAError(cudaMalloc((void**) &d_hold_1,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_hold_2,vect_size));
    HandleCUDAError(cudaMemcpy(d_A,A,mat_size,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_r_old,r_old,vect_size,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_d_old,d_old,vect_size,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_x_old,x_old,vect_size,cudaMemcpyHostToDevice));

    cudaStream_t dot_1;
    cudaStream_t dot_2;

    cudaStreamCreate(&dot_1);
    cudaStreamCreate(&dot_2);
    while(host_flag){
        MatrixVectorMult<<<blocks_per_grid,threads_per_block>>>(d_A,d_d_old,d_Ad,size);
        cudaDeviceSynchronize();
        d_Dot_Partial<<<blocks_per_grid,threads_per_block,0,dot_1>>>(d_r_old,d_r_old,d_hold_1,d_dot_partial_1,size);
        d_Dot_Partial<<<blocks_per_grid,threads_per_block,0,dot_2>>>(d_r_old,d_r_old,d_hold_1,d_dot_partial_1,size);
        cudaStreamSynchronize(dot_1);
        cudaStreamSynchronize(dot_2);
        d_Commit_Dot<<<1,blocks_per_grid,0,dot_1>>>(d_dot_partial_1,temp_1,flag);
        d_Commit_Dot<<<1,blocks_per_grid,0,dot_2>>>(d_dot_partial_2,temp_2,flag_2);
        cudaStreamSynchronize(dot_1);
        cudaStreamSynchronize(dot_2);
        comp_lamba<<<1,1,0,dot_1>>>(temp_1,temp_2,d_lambda,1,1);
        comp_lamba<<<1,1,0,dot_1>>>(temp_1,temp_2,d_lambda,1,0);
        cudaStreamSynchronize(dot_1);
        cudaStreamSynchronize(dot_2);
    }


}