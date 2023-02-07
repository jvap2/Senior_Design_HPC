#include "ZNorm.h"

__global__ void ThreadMean(float* g_A, float* g_mean, const int ny, const int nx)
{
	//fetch the global index, which will be used to access the necessary column
	int idx= threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < nx) {
		//check that in index is less than the number of columns
		float temp = 0;//Use an automatic variable to store the sum of elements within a column
		float* threadaddress = g_A + idx;//Point to the first element of a column with threadaddress
		for (int i = 0; i < ny;i++) {
			temp += threadaddress[i * nx];//Iterate through a column and sum up the entries
		}
		g_mean[idx] = temp/ny;//Solve for the mean
	}
}

__global__ void ThreadStdDev(float* g_A, float* g_mean, float* g_stddev, const int ny, const int nx)
{
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);//Fetch the global index in order to point to the start of a column
	if (idx < nx) {
		//check that the index is not greater than the number of columns
		float temp = 0;//Use this automatic variable to store the sum of the elements in a column
		float* threadaddress_A = g_A + idx;//Point to the first element in a column 
		float mean = g_mean[idx];//store the mean of a column in mean
		for (int i = 0; i < ny;i++) {
			temp += powf((threadaddress_A[i * nx]-mean),2.0f);//Iterate through the elements in a column, completing the summation portion of the stddev equation
		}
		g_stddev[idx] = sqrtf(temp / (ny-1));//Complete the standard deviation

	}
	
}

__global__ void NormMat_Naive(float* g_A, float* g_A_norm,float* g_mean, float* g_stddev, const int ny, const int nx) {

	int idx = threadIdx.x + (blockDim.x * blockIdx.x);//fetch the global index to access a specified column
	if (idx < nx) {
		float* threadaddress = g_A + idx;//point to the first element of a column in our input matrix
		float* norm_threadaddress = g_A_norm + idx;//point to the first element of a column in our normalized matrix
		float mu = g_mean[idx];//fetch the proper mean value corresponding to a specified column
		float sdev = g_stddev[idx];//fetch the proper stddev values corresponding to a specified column
		for (int i = 0; i < ny;i++) {
			//Iterate through the column and normalize each value within the column
			norm_threadaddress[i*nx] = (threadaddress[i * nx]-mu)/sdev;
		}
	}
}

//Helper function for implementing GPU matrix column mean and sample standard deviation computations with each thread computing a column mean and standard deviation
__host__ void gpuThreadHelper(float* h_A, float* h_A_Norm,float* h_mean, float* ref_mean, float* h_stddev, float* ref_stddev, float* norm_ref, const int ny, const int nx)
{
	float ElapsedTime{};
	//Global memory pointer to the data matrix
	float* d_A{};
	float* d_A_Norm{};
	//Memory size of the matrix data in bytes
	const int MatrixSizeInBytes = ny * nx * sizeof(float);
	const int Row_Bytes = nx * sizeof(float);
	//GPU global memory pointer to the mean vector
	float* d_Mean{};
	//GPU global memory pointer to the sample standard deviation vector
	float* d_Stddev{};
	//execution configuration parameters
	int threads_per_block = 256;
	int blocks_per_grid = (nx + threads_per_block - 1) / threads_per_block;
	//Allocate memory for matrix and the mean and standard deviation
	if (!HandleCUDAError(cudaMalloc((void**)&d_Mean, Row_Bytes))) {
		cout << "Cannot allocate the mean on the device" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_Stddev, Row_Bytes))) {
		cout << "Cannot allocate the standard deviation on the device" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes))) {
		cout << "Cannot allocate the matrix on the device" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_A_Norm, MatrixSizeInBytes))) {
		cout << "Cannot allocate the matrix on the device" << endl;
	}
	//Copy memory for the matrix
	if (!HandleCUDAError(cudaMemcpy(d_A,h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice))) {
		cout << "Cannot copy the matrix on the device" << endl;
	}
	//Launch the kernel for the mean
	//Create events to record the execution time of the kernel
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
	//Record the start of the execution
	HandleCUDAError(cudaEventRecord(start, 0));
	//Launch the kernel
	ThreadMean << <blocks_per_grid, threads_per_block >> > (d_A, d_Mean, ny, nx);
	cudaDeviceSynchronize();//Synchronize the host and device
	//Record the stop of the execution
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
	cout << "Exec time for the mean: " << ElapsedTime << " ms" << endl;
	//Launch the kernel for the standard deviation
	//Create events for the execution of standard deviation
	cudaEvent_t start_stddev, stop_stddev;
	HandleCUDAError(cudaEventCreate(&start_stddev));
	HandleCUDAError(cudaEventCreate(&stop_stddev));
	if (!HandleCUDAError(cudaEventRecord(start_stddev, 0))) {
		cout << "Unable to Record the start event" << endl;
	}//Record the start of execution
	ThreadStdDev << <blocks_per_grid, threads_per_block >> > (d_A, d_Mean, d_Stddev, ny, nx);
	cudaDeviceSynchronize();//Sync host and device
	//Record the stop of execution
	if (!HandleCUDAError(cudaEventRecord(stop_stddev, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	if (!HandleCUDAError(cudaEventSynchronize(stop_stddev))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//store the elapsed time from the execution of the kernel
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start_stddev, stop_stddev))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Exec time for the reduction for standard deviation: " << ElapsedTime << " ms" << endl;
	//Create events to record the execution time of the normalization
	cudaEvent_t start_norm, stop_norm;
	HandleCUDAError(cudaEventCreate(&start_norm));
	HandleCUDAError(cudaEventCreate(&stop_norm));
	if (!HandleCUDAError(cudaEventRecord(start_norm, 0))) {
		cout << "Unable to Record the start event" << endl;
	}//Record the start of execution
	//Launch the Normalization Kernel
	NormMat_Naive << <blocks_per_grid, threads_per_block >> > (d_A, d_A_Norm, d_Mean, d_Stddev, ny, nx);
	if (!HandleCUDAError(cudaEventRecord(stop_norm, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}//Record the stop of the execution
	if (!HandleCUDAError(cudaEventSynchronize(stop_norm))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start_norm, stop_norm))) {
		cout << "Unable to find elapsed time between events" << endl;
	}//Store the execution time 
	cout << "Exec time for the normalization of the input matrix: " << ElapsedTime << " ms" << endl;
	//Copy the mean vector from the device to the host
	if (!HandleCUDAError(cudaMemcpy(h_mean, d_Mean, Row_Bytes, cudaMemcpyDeviceToHost))) {
		cout << "Cannot copy the mean back on the host" << endl;
	}
	//Copy the standard deviation vector from the device to the host
	if (!HandleCUDAError(cudaMemcpy(h_stddev, d_Stddev, Row_Bytes, cudaMemcpyDeviceToHost))) {
		cout << "Cannot copy the mean back on the host" << endl;
	}
	//Copy the normalized matrix from the device to the host
	if (!HandleCUDAError(cudaMemcpy(h_A_Norm, d_A_Norm, MatrixSizeInBytes, cudaMemcpyDeviceToHost))) {
		cout << "Cannot copy the mean back on the host" << endl;
	}
	//Verify and display the final results
	cout << "Verify the standard deviation" << endl;
	string name_std = "Standard Deviation";
	Verification(name_std, ref_stddev, h_stddev, nx);
	Display(name_std, h_stddev, nx);
	cout << "Verify the mean" << endl;
	string name_mean = "Mean";
	Verification(name_mean, ref_mean, h_mean, nx);
	Display(name_std, h_mean, nx);
	cout << "Verify the normalization" << endl;
	Verification("Norm_mat", norm_ref, h_A_Norm, ny * nx);
	Display("GPU Naive Normalized Matrix", h_A_Norm, ny * nx);
	Display("CPU Normalized Matrix", norm_ref, ny * nx);

	//Destroy all of the events which were created
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));
	HandleCUDAError(cudaEventDestroy(start_stddev));
	HandleCUDAError(cudaEventDestroy(stop_stddev));
	HandleCUDAError(cudaEventDestroy(start_norm));
	HandleCUDAError(cudaEventDestroy(stop_norm));

	//Free the global memory on the device
	if (!HandleCUDAError(cudaFree(d_A))) {
		cout << "Cannot free d_A" << endl;
	}
	if (!HandleCUDAError(cudaFree(d_A_Norm))) {
		cout << "Cannot free d_A" << endl;
	}
	if (!HandleCUDAError(cudaFree(d_Mean))) {
		cout << "Cannot free d_Mean" << endl;
	}
	if (!HandleCUDAError(cudaFree(d_Stddev))) {
		cout << "Cannot free d_Stddev" << endl;
	}
	if(!HandleCUDAError(cudaDeviceReset())) {
		cout << "Cannot reset the device";
	}
}
