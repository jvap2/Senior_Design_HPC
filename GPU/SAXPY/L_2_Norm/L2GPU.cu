#include "GPUErrors.h"
#include "L2.h"


__global__ void d_L_2_Partial_Reduction(float* in, float* hold_vect, int size){
    int idx=threadIdx.x+(blockDim.x*blockIdx.x);
    int tid=threadIdx.x;
    if (idx>=size){
        return;
    }
    hold_vect[idx]=in[idx]*in[idx];
    __syncthreads();
    float* blockAddr=hold_vect+(blockDim.x+blockIdx.x);


}