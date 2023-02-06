#include "SVD.h"
#include "GPUErrors.h"

#define BX 16
#define BY 16

__global__ void Bidiag(float* A, float* p, float* p_2, float* res_1, float* res_2, float* v_1, float* v_2, int ny, int nx, int k){
    int col=threadIdx.x+(blockDim.x*blockIdx.x);
    int row = threadIdx.y+(blockDim.y*blockIdx.y);
}



__host__ void Bidiag_Helper(float* A, float* ref, float* p, float* p_2, float* res_1, float* res_2, float* v_1, float* v_2, int ny, int nx){

}