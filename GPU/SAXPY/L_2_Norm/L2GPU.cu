#include "GPUErrors.h"
#include "L2.h"


__global__ void d_L_2_Partial_Reduction(float* in, float* hold_vect,float* g_PartialSum, int size){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x);
    int tid=threadIdx.x;
    if (idx>=size){
        return;
    }
    hold_vect[idx]=in[idx]*in[idx];
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


__host__ void L_2_Helper(float* vector, float* ref,float* mu_GPU, int size){
    float* d_v;
    float* d_p_sum;
    float* d_hold_sum;
    float* d_mu;
    int vec_size=size*sizeof(float);
    int threads_p_block=256;
    int blocks_per_grid=size/threads_p_block+1;
    if(blocks_per_grid>1024){
        return;
    }
    int P_sum_size=blocks_per_grid*sizeof(float);
    HandleCUDAError(cudaMalloc((void**)&d_v,vec_size));
    HandleCUDAError(cudaMalloc((void**)&d_p_sum,P_sum_size));
    HandleCUDAError(cudaMalloc((void**)&d_hold_sum, vec_size));
    HandleCUDAError(cudaMalloc((void**)&d_mu,sizeof(float)));
    HandleCUDAError(cudaMemcpy(d_v,vector,vec_size,cudaMemcpyHostToDevice));
    d_L_2_Partial_Reduction<<<blocks_per_grid,threads_p_block>>>(d_v,d_hold_sum,d_p_sum,size);
    cudaDeviceSynchronize();
    d_Commit_L_2<<<1,blocks_per_grid>>>(d_p_sum,d_mu);
    HandleCUDAError(cudaMemcpy(mu_GPU,d_mu,sizeof(float),cudaMemcpyDeviceToHost));
    Verify(mu_GPU,ref);
    HandleCUDAError(cudaFree(d_v));
    HandleCUDAError(cudaFree(d_p_sum));
    HandleCUDAError(cudaFree(d_hold_sum));
    HandleCUDAError(cudaFree(d_mu));
    cudaDeviceReset();
}