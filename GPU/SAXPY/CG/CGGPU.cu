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