#include "GPUErrors.h"
#include "Dot.h"


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

__global__ void d_Commit_Dot(float* g_Partial_Sum, float* dot){
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
    }
}

__host__ void Dot_Helper(float* vect_1, float* vect_2, float* ref, float* dot_GPU, int size){
    float* d_v_1;
    float* d_v_2;
    float* d_hold;
    float* d_Partial;
    float* d_dot;
    int vect_size=sizeof(float)*size;
    int threads_per_block=256;
    int blocks_per_grid=size/threads_per_block+1;
    int p_sum_size=sizeof(float)*blocks_per_grid;
    HandleCUDAError(cudaMalloc((void**)&d_v_1, vect_size));
    HandleCUDAError(cudaMalloc((void**)&d_v_2, vect_size));
    HandleCUDAError(cudaMalloc((void**)&d_hold,vect_size));
    HandleCUDAError(cudaMalloc((void**)&d_Partial,p_sum_size));
    HandleCUDAError(cudaMalloc((void**)&d_dot, sizeof(float)));

    HandleCUDAError(cudaMemcpy(d_v_1,vect_1,vect_size,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_v_2,vect_2,vect_size,cudaMemcpyHostToDevice));

    d_Dot_Partial<<<blocks_per_grid,threads_per_block>>>(d_v_1,d_v_2,d_hold,d_Partial,size);
    cudaDeviceSynchronize();
    d_Commit_Dot<<<1,blocks_per_grid>>>(d_Partial,d_dot);
    cudaDeviceSynchronize();

    HandleCUDAError(cudaMemcpy(dot_GPU,d_dot,sizeof(float),cudaMemcpyDeviceToHost));
    Verify(*(dot_GPU),*(ref));

    HandleCUDAError(cudaFree(d_dot));
    HandleCUDAError(cudaFree(d_v_1));
    HandleCUDAError(cudaFree(d_v_2));
    HandleCUDAError(cudaFree(d_hold));
    HandleCUDAError(cudaFree(d_Partial));
    
    cudaDeviceReset();

}