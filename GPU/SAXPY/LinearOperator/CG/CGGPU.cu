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

__global__ void d_Const_Vect_Mult(float* vect, float* out, float* scalar, int size){
    int idx =threadIdx.x+(blockDim.x*blockIdx.x);
    if(idx<size){
        out[idx]=(*scalar)*vect[idx];
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
            int idx=abs(row-k);
			fSum += g_Matrix[idx] * g_V[k];//Here we are dotting the row of g_matrix(corresponding to the index of each thread) with g_V
		}
		g_P[row] = fSum;//We now assign the row_th entry of g_P the value fSum, i.e., our dot product
	}
}

__global__ void comp_lamba(float* in, float* in_2, float* out,int size, int flag){
    int idx =threadIdx.x+(blockDim.x*blockIdx.x);
    if(idx<size){
        if(flag==1){
            *out=*in/(*in_2);
        }
        else{
            *out=-*in/(*in_2);
        }
    }

}

__global__ void Copy(float* in, float* out, int size){
    int idx=threadIdx.x+(blockIdx.x*blockDim.x);
    if (idx<size){
        out[idx]=in[idx];
    }
}


__host__ void CG_Helper(float* A, float* ref, float* r, float* r_old, float* d, float* d_old, float* x, float* x_old, float beta, float lamdba, int size, int iter){
    float ElapsedTime{}, Elapsed_hd{}, Elapsed_dh{};
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
    int threads_per_block=128;
    int blocks_per_grid=(size/threads_per_block)+1;
    int vect_size=size*sizeof(float);
    int var_size=sizeof(float);
    int p_sum_size=sizeof(float)*blocks_per_grid;
    HandleCUDAError(cudaMalloc((void**) &d_A,vect_size));
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
    HandleCUDAError(cudaMalloc((void**) &d_Ad,vect_size));
    HandleCUDAError(cudaMalloc((void**) &lamd_d,vect_size));
    HandleCUDAError(cudaMalloc((void**) &beta_d,vect_size));
    HandleCUDAError(cudaMalloc((void**) &lambd_AD,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_dot_partial_1,p_sum_size));
    HandleCUDAError(cudaMalloc((void**) &d_dot_partial_2,p_sum_size));
    HandleCUDAError(cudaMalloc((void**) &d_hold_1,vect_size));
    HandleCUDAError(cudaMalloc((void**) &d_hold_2,vect_size));

    cudaStream_t dot_1;
    cudaStream_t dot_2;
    cudaStream_t copy_1;
    cudaStream_t copy_2;
    cudaStream_t copy_3;
    cudaStreamCreate(&dot_1);
    cudaStreamCreate(&dot_2);
    cudaStreamCreate(&copy_1);
    cudaStreamCreate(&copy_2);
    cudaStreamCreate(&copy_3);
    cudaEvent_t start, stop;
    cudaEvent_t start_dh, stop_dh;
    cudaEvent_t start_hd, stop_hd;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
    HandleCUDAError(cudaEventCreate(&start_dh));
    HandleCUDAError(cudaEventCreate(&stop_dh));
    HandleCUDAError(cudaEventCreate(&start_hd));
    HandleCUDAError(cudaEventCreate(&stop_hd));
    HandleCUDAError(cudaEventRecord(start_hd,0));
    HandleCUDAError(cudaMemcpy(d_A,A,vect_size,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_r_old,r_old,vect_size,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_d_old,d_old,vect_size,cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_x_old,x_old,vect_size,cudaMemcpyHostToDevice));
    if (!HandleCUDAError(cudaEventRecord(stop_hd, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop_hd))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Save the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&Elapsed_hd, start_hd, stop_hd))) {
		cout << "Unable to find elapsed time between events" << endl;
	}

    HandleCUDAError(cudaEventRecord(start,0));
    for(int i{}; i<iter; i++){
        MatrixVectorMult<<<blocks_per_grid,threads_per_block>>>(d_A,d_d_old,d_Ad,size);
        cudaDeviceSynchronize();
        d_Dot_Partial<<<blocks_per_grid,threads_per_block,0,dot_1>>>(d_r_old,d_r_old,d_hold_1,d_dot_partial_1,size);
        d_Dot_Partial<<<blocks_per_grid,threads_per_block,0,dot_2>>>(d_d_old,d_Ad,d_hold_2,d_dot_partial_2,size);
        cudaStreamSynchronize(dot_1);
        cudaStreamSynchronize(dot_2);
        d_Commit_Dot<<<1,blocks_per_grid,0,dot_1>>>(d_dot_partial_1,temp_1);
        d_Commit_Dot<<<1,blocks_per_grid,0,dot_2>>>(d_dot_partial_2,temp_2);
        cudaStreamSynchronize(dot_1);
        cudaStreamSynchronize(dot_2);

        comp_lamba<<<1,1,0,dot_1>>>(temp_1,temp_2,d_lambda,1,1);
        comp_lamba<<<1,1,0,dot_2>>>(temp_1,temp_2,d_neg_lambda,1,0);
        cudaStreamSynchronize(dot_1);
        cudaStreamSynchronize(dot_2);
        //Need to do scalar mult
        d_Const_Vect_Mult<<<blocks_per_grid,threads_per_block,0,dot_1>>>(d_d_old,lamd_d,d_lambda,size);
        d_Const_Vect_Mult<<<blocks_per_grid,threads_per_block,0,dot_2>>>(d_Ad,lambd_AD,d_neg_lambda,size);
        cudaStreamSynchronize(dot_1);
        cudaStreamSynchronize(dot_2);
        VectAdd<<<blocks_per_grid,threads_per_block,0,dot_1>>>(d_x_old,lamd_d,d_x,size);
        VectAdd<<<blocks_per_grid,threads_per_block,0,dot_2>>>(d_r_old,lambd_AD,d_r,size);
        cudaStreamSynchronize(dot_1);
        cudaStreamSynchronize(dot_2);

        d_Dot_Partial<<<blocks_per_grid,threads_per_block>>>(d_r,d_r,d_hold_2,d_dot_partial_2,size);
        cudaDeviceSynchronize();

        d_Commit_Dot<<<1,blocks_per_grid>>>(d_dot_partial_2,temp_2);
        cudaDeviceSynchronize();

        comp_lamba<<<1,1>>>(temp_2,temp_1,d_beta,1,1);
        cudaDeviceSynchronize();

        d_Const_Vect_Mult<<<blocks_per_grid,threads_per_block>>>(d_d_old,beta_d,d_beta,size);
        cudaDeviceSynchronize();

        VectAdd<<<blocks_per_grid,threads_per_block>>>(d_r,beta_d,d_d,size);
        cudaDeviceSynchronize();

        Copy<<<blocks_per_grid,threads_per_block,0,copy_1>>>(d_d,d_d_old,size);
        Copy<<<blocks_per_grid,threads_per_block,0,copy_2>>>(d_r,d_r_old,size);
        Copy<<<blocks_per_grid,threads_per_block,0,copy_3>>>(d_x,d_x_old,size);
        cudaStreamSynchronize(copy_1);
        cudaStreamSynchronize(copy_2);
        cudaStreamSynchronize(copy_3);

    }
    cudaStreamDestroy(dot_1);
    cudaStreamDestroy(dot_2);
    cudaStreamDestroy(copy_1);
    cudaStreamDestroy(copy_2);
    cudaStreamDestroy(copy_3);
    if (!HandleCUDAError(cudaEventRecord(stop, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Save the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
    HandleCUDAError(cudaEventRecord(start_dh,0));
    HandleCUDAError(cudaMemcpy(x,d_x,vect_size,cudaMemcpyDeviceToHost));
    if (!HandleCUDAError(cudaEventRecord(stop_dh, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop_dh))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Save the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&Elapsed_dh, start_dh, stop_dh))) {
		cout << "Unable to find elapsed time between events" << endl;
	}

    float total_time=Elapsed_dh+Elapsed_hd;
    float bytes_transferred=(5*vect_size)*1.0f;
    float throughput=(bytes_transferred*1e-6)/(total_time);
    cout<< "GPU CG Memory elapsed time: "<<total_time<< " ms"<<endl;
    cout<< "GPU CG Exec elapsed time: "<<ElapsedTime<< " ms"<<endl;
    cout<< "GPU CG total elapsed time: "<<ElapsedTime+total_time<< " ms"<<endl;
    cout<<"GPU throughput: "<<throughput<< "GB/s"<<endl;

    Verify(x,ref,size);
    HandleCUDAError(cudaFree(d_A));
    HandleCUDAError(cudaFree(d_r));
    HandleCUDAError(cudaFree(d_r_old));
    HandleCUDAError(cudaFree(d_d));
    HandleCUDAError(cudaFree(d_d_old));
    HandleCUDAError(cudaFree(d_x));
    HandleCUDAError(cudaFree(d_x_old));
    HandleCUDAError(cudaFree(d_beta));
    HandleCUDAError(cudaFree(d_neg_lambda));
    HandleCUDAError(cudaFree(d_lambda));
    HandleCUDAError(cudaFree(temp_1));
    HandleCUDAError(cudaFree(temp_2));
    HandleCUDAError(cudaFree(d_hold_1));
    HandleCUDAError(cudaFree(d_hold_2));
    HandleCUDAError(cudaFree(d_Ad));
    HandleCUDAError(cudaFree(d_dot_partial_1));
    HandleCUDAError(cudaFree(d_dot_partial_2));
    HandleCUDAError(cudaFree(lambd_AD));
    HandleCUDAError(cudaFree(lamd_d));
    HandleCUDAError(cudaFree(beta_d));
    cudaDeviceReset();


}