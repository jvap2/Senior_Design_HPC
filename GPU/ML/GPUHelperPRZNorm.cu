#include "ZNorm.h"
#define ST_DIMX 16
#define ST_DIMY 16

__global__ void PartialReduction(float* g_Vector, float* g_PartialSum, const int size)
{
	int tid = threadIdx.x;//Obtain the local index for the section of the vector given to each block

	//Compute the global thread index
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);//Obtain the global index for ensuring the threads are within the bounds of the size of the matrix
	float* blockAddress = g_Vector + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block

	if (idx >= size)
	{
		return;//Leave the kernel if the global index of our thread exceeds the size of the vector being processed
	}
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
		g_PartialSum[blockIdx.x] = blockAddress[0];//thread 0 will store the partial thread of the block based on the in place methodology
		//Hence, we store the first element of blockAddress in partial sum of each blockIdx.x 
	}
}

//Kernel code for transposing the data matrix
__global__ void Transpose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx)
{
	//Declare static shared memory 
	__shared__ float tile[ST_DIMY][ST_DIMX];//tile of 16 by 16, transfer data of block into the shared memory

	//Coordinates in original matrix
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	//Linear global memory index address in the original matrix
	unsigned int ti = iy * nx + ix;//This means the row major access

	//Lines 50-52 are designed to get a 2 dimension index of the elements in our shared memory tile
	unsigned int bidx = threadIdx.x + (threadIdx.y * blockDim.x);//This gives the linear index of our element relative to a tile
	unsigned int irow = bidx / blockDim.y;//This gives the row in the tile
	unsigned int icol = bidx % blockDim.x;//This gives us the column in the tile

	//Coordinates in transpose matrix
	ix = icol + (blockIdx.y * blockDim.y);//Gives the new column for the transpose matrix
	iy = irow + (blockIdx.x * blockDim.x);//Gives the new row for the transpose matrix

	//linear global memory index address in the transpose matrix
	unsigned int to = iy * ny + ix;

	if (ix < ny && iy < nx)
	{
		//Load the data from the original matrix into the tile on the shared memory
		tile[threadIdx.y][threadIdx.x] = g_Matrix[ti];
		__syncthreads();//Wait until the data has been loaded into the shared matrix by every thread in a block sharing this memory
		g_MatrixTranspose[to] = tile[icol][irow];//Transpose the elements within a tile and load it to transpose 
	}
	//Effectively what is happening here is we are transposing the tiles as if they are elements, and then once we load it into the transpose matrix by tranposing the elements within the tile
}

__global__ void PartialReduction_STDDEV(float* g_Vector, float* g_PartialSum, float g_mean, const int size)
{
	int tid = threadIdx.x;//Obtain the local index for the section of the vector given to each block

	//Compute the global thread index
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); //Obtain the global index for ensuring the threads are within the bounds of the size of the matrix
	float* blockAddress = g_Vector + (blockIdx.x * blockDim.x);//Use this to point to the start of the vector allocated to each block

	if (idx >= size)
	{
		return;//Leave the kernel if the global index of our thread exceeds the size of the vector being processed
	}
	blockAddress[tid] = powf(blockAddress[tid] - g_mean, 2.0f);//Update all of the block address values with (bAdd[tid]-mean)^2 and finish the remainder of stddev computation on the CPU
	__syncthreads();//Wait until all of the blockAddress values are updated
	//Perform the interleaved reduction, used to reduce divergence.
	//Start adding elements blockDim.x apart, store in place and then half the stride and continue until stride=1
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		//tid<stride ensures we do not try to access memory past the vector allocated to the block
		//tid+stride<size allows for vector sizes less than blockDim
		if (tid < stride && tid+stride<size)
		{
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
	if (tid == 0)
	{
		g_PartialSum[blockIdx.x] = blockAddress[0];//thread 0 will store the partial thread of the block based on the in place methodology
		//Hence, we store the first element of blockAddress in partial sum of each blockIdx.x 
	}

}

__global__ void NormalizeMatrix(float* g_Vector, float* g_Vector_Norm, float mean, float stddev, const int size) {
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);//Get the global index of our element we are going to normalize
	if (idx >= size) {
		return;//If the global index exceeds the presubscribed size, exit the kernel
	}
	g_Vector_Norm[idx] = (g_Vector[idx] - mean) / stddev;//Create a normalized matrix with the z score equation
}


//
//Helper function for implementing GPU matrix mean computation with each block computing a column mean, this one using 4 streams
__host__ void gpuPRMeanHelper(float* h_A, float* h_A_Norm, float* h_mean, float* ref, float*h_stddev,float* ref_stddev, float* norm_ref, const int ny, const int nx)
{
	//GPU global memory pointer to the matrix
	float* d_A{};//Pointer for the A matrix to be stored on the device
	float* d_A_Norm{};//Pointer for the A_norm matrix to be stored on the device
	//Global memory pointer to the transpose of the data matrix
	float* d_transpose{};
	//Memory size of the matrix in bytes
	const int MatrixSizeInBytes = ny * nx * sizeof(float);
	//global memory pointer to store the partial sums
	float* d_PartialSums{};
	//global memory pointer to store the partial sums for standard deviation
	float* d_PartialSums_stddev{};
	//Host memory pointer to store the partial sums
	float* h_PartialSums{};
	//Host memory pointer to store the partial sums for the standard deviation
	float* h_PartialSums_stddev{};
	//Create the execution configuration parameters for the transpose
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);//Create the necessary blocks per grid
	/*--------------------------------------------------------------------------------------------------------------------*/
	/*----------------------------------- Execution Configuration Parameters for Partial Sum------------------------------*/
	int threads_per_block = 256;//Declare the number of threads/block as specified in the instructions
	int blocks_per_grid =ceil(1.0*ny/threads_per_block);//find the number of blocks that will allow for completing
	/*--------------------------------------------------------------------------------------------------------------------*/

	//the partial sum of the elements in a row- will give us blocks_per_grid number of partial sums that must later be combined on the CPU
	cout << "2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;
	cout << "-------------------------------------------" << endl;
	cout << "1D dimensions for the partial sum" << endl;
	cout << "Threads_per_block\t" << threads_per_block << endl;
	cout << "blocks_per_grid\t" << blocks_per_grid << endl;
	//Create variable for memory sixze of partial sums
	int size_psum = nx * blocks_per_grid * sizeof(float);
	//Allocate the necessary memory on the device 
	if (!HandleCUDAError(cudaHostAlloc((void**)&h_PartialSums, size_psum, cudaHostAllocDefault))) {
		cout << "Unable to allocate pinned memory for the h_PartialSums" << endl;
	}
	if (!HandleCUDAError(cudaHostAlloc((void**)&h_PartialSums_stddev, size_psum, cudaHostAllocDefault))) {
		cout << "Unable to allocate pinned memory for the h_PartialSums" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_A" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_A_Norm, MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_A" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_transpose,MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_transpose" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_PartialSums, size_psum))) {
		cout << "Cannot Allocate the memory for d_PartialSums" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_PartialSums_stddev, size_psum))) {
		cout << "Cannot Allocate the memory for d_PartialSums" << endl;
	}
	//Copy the input matrix to the device in order to perfrom the transpose
	if (!HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice))) {
		cout << "Cannot copy the matrix over" << endl;
	}
	//Call the tranpose kernel
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	//Synchronize the host and the device
	cudaDeviceSynchronize();
	float ElapsedTime{};
	//Create cudaEvent_t variables to record the time
	cudaEvent_t start, stop;
	//Create the start and stop event to record exec time
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
	//Declear stream variables, 4 of them for this kernel
	cudaStream_t stream_1, stream_2, stream_3, stream_4;
	//Create the streams to utilize when launching the kernels
	if (!HandleCUDAError(cudaStreamCreate(&stream_1))) {
		cout << "Cannot Create stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_2))) {
		cout << "Cannot Create stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_3))) {
		cout << "Cannot Create stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_4))) {
		cout << "Cannot Create stream_4" << endl;
	}
	if (!HandleCUDAError(cudaEventRecord(start, 0))) {
		cout << "Unable to Record the start event" << endl;
	}
	for (unsigned int i = 0;i <nx;i += 4) {
		/* Within this loop, we are iterating through PartialReduction kernel launches
		We are going to allocate a row to each kernel launch, and in this for loop we are working with 4 rows at a time with four differnet kernel launches
		thi should be noted by the calls such as ((d_transpose + (i * ny)), ((d_transpose + ((i + 1) * ny)) etc. where we are pointing to the ith row and (i+1)th row respecitvely */
		PartialReduction << <blocks_per_grid, threads_per_block,0,stream_1 >> > ((d_transpose + (i * ny)), (d_PartialSums + (i * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block,0,stream_2 >> > ((d_transpose + ((i + 1) * ny)), (d_PartialSums + ((i + 1) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block,0,stream_3 >> > ((d_transpose + ((i + 2) * ny)), (d_PartialSums + ((i + 2) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block,0,stream_4 >> > ((d_transpose + ((i + 3) * ny)), (d_PartialSums + ((i + 3) * blocks_per_grid)), ny);
		/* Here, we are asynchronously copying the results back to the ith, (i+1)th etc row of d_PartialSums to the same row of h_PartialSums, a section of pinned*/
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + (i * blocks_per_grid)), (d_PartialSums + (i * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_1));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 1) * blocks_per_grid)), (d_PartialSums + ((i + 1) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_2));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 2) * blocks_per_grid)), (d_PartialSums + ((i + 2) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_3));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 3) * blocks_per_grid)), (d_PartialSums + ((i + 3) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_4));
	}
	/* After completion of the loop we need to ensure the synchronize the streams before we can proceed as shown*/
	if (!HandleCUDAError(cudaStreamSynchronize(stream_1))) {
		cout << "Unable to perform stream synch with stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_2))) {
		cout << "Unable to perform stream synch with stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_3))) {
		cout << "Unable to perform stream synch with stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_4))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Here, once the streams are synchronized, can record the stop event for the execution time
	if (!HandleCUDAError(cudaEventRecord(stop,0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//We must synchronize the event as well otherwise, this will not function
	if (!HandleCUDAError(cudaEventSynchronize(stop))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//We are now storing the difference between stop and start into elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Total time for stream partial sums: " << ElapsedTime <<" ms"<< endl;
	//Below, we are now post processing the h_PartialSums to find the means
	float* vect_temp;
	for (int i = 0; i < nx;i++) {
		//Here we are iterating through the rows of the partial sums matrix with this outermost loop
		float temp = 0;//Use this value to store the sum before dividing for the mean
		vect_temp = (h_PartialSums + (i * blocks_per_grid));//Point to a new row of the partialsums matrix
		for (int j = 0;j < blocks_per_grid;j++) {
			//Iterate through the columns of our vector
			temp += *(vect_temp + j);
		}
		//Now, we store temp, divided by the number of rows to h_mean[i]
		h_mean[i] = temp/ny;
	}
	//Verify the correct calculations and display
	Verification("PR Mean",ref,h_mean, nx);
	Display("h_Mean", h_mean, nx);
	Display("ref", ref, nx);
	//Create new events for recording the execution time of the standard deviation
	cudaEvent_t start_stddev, stop_stddev;
	HandleCUDAError(cudaEventCreate(&start_stddev));
	HandleCUDAError(cudaEventCreate(&stop_stddev));
	//Record the start of the execution
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	cudaDeviceSynchronize();
	if (!HandleCUDAError(cudaEventRecord(start_stddev, 0))) {
		cout << "Unable to Record the start event" << endl;
	}
	//Calculate the standard deviation now using some form of reduction
	for (unsigned int i = 0;i < nx;i += 4) {
		/* Within this loop, we are iterating through PartialReduction_STDDEV kernel launches
		We are going to allocate a row to each kernel launch, and in this for loop we are working with 4 rows at a time with four differnet kernel launches
		should be noted by the calls such as ((d_transpose + (i * ny)), ((d_transpose + ((i + 1) * ny)) etc. where we are pointing to the ith row and (i+1)th row respecitvely */
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_1 >> > ((d_transpose + (i * ny)), (d_PartialSums_stddev + (i * blocks_per_grid)),h_mean[i], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_2 >> > ((d_transpose + ((i + 1) * ny)), (d_PartialSums_stddev + ((i + 1) * blocks_per_grid)), h_mean[i+1], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_3 >> > ((d_transpose + ((i + 2) * ny)), (d_PartialSums_stddev + ((i + 2) * blocks_per_grid)), h_mean[i+2], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_4 >> > ((d_transpose + ((i + 3) * ny)), (d_PartialSums_stddev + ((i + 3) * blocks_per_grid)), h_mean[i+3], ny);
		/* Here, we are asynchronously copying the results back to the ith, (i+1)th etc row of d_PartialSums_stddev to the same row of h_PartialSums_stddev, a section of pinned*/
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + (i * blocks_per_grid)), (d_PartialSums_stddev + (i * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_1));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 1) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 1) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_2));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 2) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 2) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_3));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 3) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 3) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_4));
	}
	/* After completion of the loop we need to ensure the synchronize the streams before we can proceed as shown*/
	if (!HandleCUDAError(cudaStreamSynchronize(stream_1))) {
		cout << "Unable to perform stream synch with stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_2))) {
		cout << "Unable to perform stream synch with stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_3))) {
		cout << "Unable to perform stream synch with stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_4))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Record the stop of our event now that the streams are synchronize
	if (!HandleCUDAError(cudaEventRecord(stop_stddev, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the the stop for the standard deviation
	if (!HandleCUDAError(cudaEventSynchronize(stop_stddev))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Store stop_stddev-start_stddev to ElapsedTime
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start_stddev, stop_stddev))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Exec time for the reduction for standard deviation: " << ElapsedTime << " ms" << endl;
	//Now, we will post process the partial sums for the standard deviation
	float* vect_temp_stddev;
	for (int i = 0;i < nx;i++) {
		//Here we are iterating through the rows of the partial sums matrix with this outermost loop
		float temp = 0;//use this to sum the rows of each partial sum before committing
		vect_temp_stddev = h_PartialSums_stddev + (i * blocks_per_grid);//point to a new row of stddev partial
		for (int j = 0;j < blocks_per_grid;j++) {
			temp += *(vect_temp_stddev + j);//sum the columns of each row
		}
		h_stddev[i] = sqrtf(temp / (ny-1));
	}
	//Verify and Display the results of the computations
	Verification("PR Stddev", ref_stddev, h_stddev, nx);
	Display("h_stddev", h_stddev, nx);
	Display("ref", ref_stddev, nx);
	//Create the events for normalizing the matrix
	cudaEvent_t start_norm, stop_norm;
	HandleCUDAError(cudaEventCreate(&start_norm));
	HandleCUDAError(cudaEventCreate(&stop_norm));
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	cudaDeviceSynchronize();
	//Record the start of the execution of streams
	if (!HandleCUDAError(cudaEventRecord(start_norm, 0))) {
		cout << "Unable to Record the start event" << endl;
	}

	//Calculate the standard deviation now using some form of reduction
	for (unsigned int i = 0;i < nx;i += 4) {		
		/* Within this loop, we are iterating through the NormalizeMatrix kernel launches
		We are going to allocate a row to each kernel launch, and in this for loop we are working with 4 rows at a time with four differnet kernel launches
		should be noted by the calls such as ((d_transpose + (i * ny)), ((d_transpose + ((i + 1) * ny)) etc. where we are pointing to the ith row and (i+1)th row respecitvely */
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_1 >> > ((d_transpose + (i * ny)), (d_A_Norm + (i * ny)), h_mean[i], h_stddev[i], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_2 >> > ((d_transpose + ((i+1) * ny)), (d_A_Norm + ((i+1) * ny)), h_mean[i+1], h_stddev[i+1], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_3 >> > ((d_transpose + ((i+2) * ny)), (d_A_Norm + ((i+2) * ny)), h_mean[i+2], h_stddev[i+2], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_4 >> > ((d_transpose + ((i+3) * ny)), (d_A_Norm + ((i+3) * ny)), h_mean[i+3], h_stddev[i+3], ny);
	}
	//Synchronize the streams after iterating through kernel launches
	if (!HandleCUDAError(cudaStreamSynchronize(stream_1))) {
		cout << "Unable to perform stream synch with stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_2))) {
		cout << "Unable to perform stream synch with stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_3))) {
		cout << "Unable to perform stream synch with stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_4))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Record the stop of the streams for the total elapsed time
	if (!HandleCUDAError(cudaEventRecord(stop_norm, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	if (!HandleCUDAError(cudaEventSynchronize(stop_norm))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Store the total elapsed time in Elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start_norm, stop_norm))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Exec time for the normalization of the input matrix: " << ElapsedTime << " ms" << endl;
	//Copy the normalized matrix from the device to the host
	if (!HandleCUDAError(cudaMemcpy(h_A_Norm, d_A_Norm, MatrixSizeInBytes, cudaMemcpyDeviceToHost))) {
		cout << "Cannot copy the normalized matrix back to the host" << endl;
	}
	//Verify the normalized matrix is correct, and display
	Verification("Normalized Matrix", norm_ref, h_A_Norm, ny*nx);
	cout << "GPU Normalized Matrix" << endl;
	Display("h_A_Norm",h_A_Norm, ny*nx);
	cout << "CPU Normalized Matrix" << endl;
	Display("norm_ref",norm_ref, ny*nx);
	cout << "Exec time for the reduction for Normalization: " << ElapsedTime << " ms" << endl;
	//Free the pinned memory on the host
	if (!HandleCUDAError(cudaFreeHost(h_PartialSums))) {
		cout << "Unable to free the h_PartialSums memory on the host" << endl;
	}
	if (!HandleCUDAError(cudaFreeHost(h_PartialSums_stddev))) {
		cout << "Unable to free the h_PartialSums memory on the host" << endl;
	}
	//Destroy the streams we have generated
	if (!HandleCUDAError(cudaStreamDestroy(stream_1))) {
		cout << "Unable to destroy stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_2))) {
		cout << "Unable to destroy stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_3))) {
		cout << "Unable to destroy stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_4))) {
		cout << "Unable to destroy stream_4" << endl;
	}
	//Destroy the events
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));
	HandleCUDAError(cudaEventDestroy(start_stddev));
	HandleCUDAError(cudaEventDestroy(stop_stddev));
	HandleCUDAError(cudaEventDestroy(start_norm));
	HandleCUDAError(cudaEventDestroy(stop_norm));
	//Free all of the memory allocated on the device
	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_A_Norm));
	HandleCUDAError(cudaFree(d_PartialSums));
	HandleCUDAError(cudaFree(d_PartialSums_stddev));
	HandleCUDAError(cudaFree(d_transpose));
	HandleCUDAError(cudaDeviceReset());
}

__host__ void gpuPRMeanHelper_SingleStream(float* h_A, float* h_A_Norm, float* h_mean, float* ref, float* h_stddev,float* ref_stddev, float* norm_ref, const int ny, const int nx)
{
	//GPU global memory pointer to the matrix
	float* d_A{};
	float* d_A_Norm{};
	//Global memory pointer to the transpose of the data matrix
	float* d_transpose{};
	//Memory size of the matrix in bytes
	const int MatrixSizeInBytes = ny * nx * sizeof(float);
	//global memory pointer to store the partial sums
	float* d_PartialSums{};
	float* d_PartialSums_stddev{};
	//Host memory pointer to store the partial sums for mean and stddev
	float* h_PartialSums{};
	float* h_PartialSums_stddev{};
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);//Create the necessary blocks per grid for the transpose
	int threads_per_block = 256;//Declare the number of threads/block as specified in the instructions for all other kernels besides the transpose
	int blocks_per_grid = ceil(1.0 * ny / threads_per_block);//find the number of blocks that will allow for completing
	//the partial sum of the elements in a row- will give us blocks_per_grid number of partial sums that must later be combined on the CPU
	cout << "2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;
	cout << "-------------------------------------------" << endl;
	cout << "1D dimensions for the partial sum" << endl;
	cout << "Threads_per_block\t" << threads_per_block << endl;
	cout << "blocks_per_grid\t" << blocks_per_grid << endl;
	int size_psum = nx * blocks_per_grid * sizeof(float);//Store a variable for the size of the partial sums matrix

	//Allocate the pinned memory for the partial sums for mean and stddev
	if (!HandleCUDAError(cudaHostAlloc((void**)&h_PartialSums, size_psum, cudaHostAllocDefault))) {
		cout << "Unable to allocate pinned memory for the h_PartialSums" << endl;
	}
	if (!HandleCUDAError(cudaHostAlloc((void**)&h_PartialSums_stddev, size_psum, cudaHostAllocDefault))) {
		cout << "Unable to allocate pinned memory for the h_PartialSums" << endl;
	}
	
	//Allocate the necessary memory on the device
	if (!HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_A" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_A_Norm, MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_A" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_transpose, MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_transpose" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_PartialSums, size_psum))) {
		cout << "Cannot Allocate the memory for d_PartialSums" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_PartialSums_stddev, size_psum))) {
		cout << "Cannot Allocate the memory for d_PartialSums" << endl;
	}
	if (!HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice))) {
		cout << "Cannot copy the matrix over" << endl;
	}
	//Launch the transpose
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	cudaDeviceSynchronize();//synchronize the host and device
	float ElapsedTime{};
	//Generate events to record the execution time
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
	HandleCUDAError(cudaEventRecord(start, 0));
	for (unsigned int i = 0;i < nx;i++) {
		/* Within this loop, we are iterating through PartialReduction kernel launches
		We are going to allocate a row to each kernel launch, and in this for loop we are working with one row each iteration with one kernel launch
		Then we copy back the row we have processed to the proper row on the host*/
		PartialReduction << <blocks_per_grid, threads_per_block>> > ((d_transpose + (i * ny)), (d_PartialSums + (i * blocks_per_grid)), ny);
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + (i * blocks_per_grid)), (d_PartialSums + (i * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost));
	}
	//We do not need to synchronize streams after completing the iteration since we are using the default stream
	//Record the stop for exec time
	if (!HandleCUDAError(cudaEventRecord(stop, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Store the execution time in the variable ElapsedTime
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Exec time for the mean: " << ElapsedTime << " ms" << endl;
	float* vect_temp;
	for (int i = 0;i < nx;i++) {
		//Iterate through the rows of the partial sum matrix
		float temp = 0;//Use this variable to store the sum of each row before dividing by ny
		vect_temp = h_PartialSums + i * blocks_per_grid;//point to the start of the new row
		for (int j = 0;j < blocks_per_grid;j++) {
			temp += *(vect_temp + j);//Sum the row
		}
		h_mean[i] = (temp / ny);
	}
	//verify and display the calculation
	Verification("PR Mean", ref, h_mean, nx);
	Display("h_Mean", h_mean, nx);
	Display("ref", ref, nx);
	//Launch the transpose
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	cudaDeviceSynchronize();
	//Calculate the standard deviation now using some form of reduction
	//Create events to record the execution time of the standard deviation on the device
	cudaEvent_t start_stddev, stop_stddev;
	HandleCUDAError(cudaEventCreate(&start_stddev));
	HandleCUDAError(cudaEventCreate(&stop_stddev));
	if (!HandleCUDAError(cudaEventRecord(start_stddev, 0))) {
		cout << "Unable to Record the start event" << endl;
	}
	for (unsigned int i = 0;i < nx;i++) {
		/* Within this loop, we are iterating through PartialReduction kernel launches
		We are going to allocate a row to each kernel launch, and in this for loop we are working with 1 row at a time with kernel launches
		we then copy back the processed data to its respective row on the host asynchronously*/
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, 0 >> > ((d_transpose + (i * ny)), (d_PartialSums_stddev + (i * blocks_per_grid)), h_mean[i], ny);
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + (i * blocks_per_grid)), (d_PartialSums_stddev + (i * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost,0));
	}
	//Record the stop of execution and synchronize it
	if (!HandleCUDAError(cudaEventRecord(stop_stddev, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	if (!HandleCUDAError(cudaEventSynchronize(stop_stddev))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Store the elapsed time
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start_stddev, stop_stddev))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Exec time for the reduction for standard deviation: " << ElapsedTime << " ms" << endl;
	float* vect_temp_stddev;
	for (int i = 0;i < nx;i++) {
		//Iterate through the rows of the partial sum
		float temp = 0;
		vect_temp_stddev = h_PartialSums_stddev + (i * blocks_per_grid);//point to the start of a new row each iteration
		for (int j = 0;j < blocks_per_grid;j++) {
			temp += *(vect_temp_stddev + j);//sum up the row
		}
		h_stddev[i] = sqrtf(temp / (ny - 1));
	}
	//Display and Verify the stddev calculations
	/*DisplayMatrix(h_PartialSums_stddev, nx, blocks_per_grid);*/
	Verification("PR Stddev", ref_stddev, h_stddev, nx);
	Display("h_stddev", h_stddev, nx);
	Display("ref", ref_stddev, nx);
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	cudaDeviceSynchronize();
	//Creat the events to record the elapsed time of the norm calculations
	cudaEvent_t start_norm, stop_norm;
	HandleCUDAError(cudaEventCreate(&start_norm));
	HandleCUDAError(cudaEventCreate(&stop_norm));
	//Record the start of the execution
	if (!HandleCUDAError(cudaEventRecord(start_norm, 0))) {
		cout << "Unable to Record the start event" << endl;
	}
	for (unsigned int i = 0;i < nx;i++) {
		/* Within this loop, we are iterating through Normalize kernel launches, and we normalize each row individually. Each iteration, we point to the ith row*/
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0,0 >> > ((d_transpose + (i * ny)), (d_A_Norm + (i * ny)), h_mean[i], h_stddev[i], ny);
	}
	//record the stop of execution and sync the event
	if (!HandleCUDAError(cudaEventRecord(stop_norm, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	if (!HandleCUDAError(cudaEventSynchronize(stop_norm))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Find the elapsed time of execution for our normalization code
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start_norm, stop_norm))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Exec time for the normalization of the input matrix: " << ElapsedTime << " ms" << endl;
	//Copy the normalized matrix from the device to the host
	if (!HandleCUDAError(cudaMemcpy(h_A_Norm, d_A_Norm, MatrixSizeInBytes, cudaMemcpyDeviceToHost))) {
		cout << "Cannot copy the normalized matrix back to the host" << endl;
	}
	//Verify and display the normalized matrix
	Verification("Normalized Matrix", norm_ref, h_A_Norm, ny * nx);
	cout << "GPU Normalized Matrix" << endl;
	Display("h_A_Norm", h_A_Norm, ny * nx);
	cout << "CPU Normalized Matrix" << endl;
	Display("norm_ref", norm_ref, ny * nx);
	cout << "Exec time for the reduction for Normalization: " << ElapsedTime << " ms" << endl;
	//Free the pinned memory
	if (!HandleCUDAError(cudaFreeHost(h_PartialSums))) {
		cout << "Unable to free the h_PartialSums memory on the host" << endl;
	}
	if (!HandleCUDAError(cudaFreeHost(h_PartialSums_stddev))) {
		cout << "Unable to free the h_PartialSums memory on the host" << endl;
	}
	//Destroy all the events and free the global device memory
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));
	HandleCUDAError(cudaEventDestroy(start_stddev));
	HandleCUDAError(cudaEventDestroy(stop_stddev));
	HandleCUDAError(cudaEventDestroy(start_norm));
	HandleCUDAError(cudaEventDestroy(stop_norm));

	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_A_Norm));
	HandleCUDAError(cudaFree(d_PartialSums));
	HandleCUDAError(cudaFree(d_PartialSums_stddev));
	HandleCUDAError(cudaFree(d_transpose));
	HandleCUDAError(cudaDeviceReset());
}

__host__ void gpuPRMeanHelper_16Stream(float* h_A, float* h_A_Norm, float* h_mean, float* ref, float* h_stddev, float* ref_stddev, float* norm_ref, const int ny, const int nx)
{
	//GPU global memory pointer to the matrix
	float* d_A{};//Pointer for the A matrix to be stored on the device
	float* d_A_Norm{};//Pointer for the A_norm matrix to be stored on the device
	//Global memory pointer to the transpose of the data matrix
	float* d_transpose{};
	//Memory size of the matrix in bytes
	const int MatrixSizeInBytes = ny * nx * sizeof(float);
	//global memory pointer to store the partial sums
	float* d_PartialSums{};
	//global memory pointer to store the partial sums for standard deviation
	float* d_PartialSums_stddev{};
	//Host memory pointer to store the partial sums
	float* h_PartialSums{};
	//Host memory pointer to store the partial sums for the standard deviation
	float* h_PartialSums_stddev{};
	//Create the execution configuration parameters for the transpose
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);//Create the necessary blocks per grid
	/*--------------------------------------------------------------------------------------------------------------------*/
	/*----------------------------------- Execution Configuration Parameters for Partial Sum------------------------------*/
	int threads_per_block = 256;//Declare the number of threads/block as specified in the instructions
	int blocks_per_grid = ceil(1.0 * ny / threads_per_block);//find the number of blocks that will allow for completing
	/*--------------------------------------------------------------------------------------------------------------------*/

	//the partial sum of the elements in a row- will give us blocks_per_grid number of partial sums that must later be combined on the CPU
	cout << "2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;
	cout << "-------------------------------------------" << endl;
	cout << "1D dimensions for the partial sum" << endl;
	cout << "Threads_per_block\t" << threads_per_block << endl;
	cout << "blocks_per_grid\t" << blocks_per_grid << endl;
	int size_psum = nx * blocks_per_grid * sizeof(float);//create variable for the size of the partial sum matrix in bytes
	//Allocate the the pinned memory
	if (!HandleCUDAError(cudaHostAlloc((void**)&h_PartialSums, size_psum, cudaHostAllocDefault))) {
		cout << "Unable to allocate pinned memory for the h_PartialSums" << endl;
	}
	if (!HandleCUDAError(cudaHostAlloc((void**)&h_PartialSums_stddev, size_psum, cudaHostAllocDefault))) {
		cout << "Unable to allocate pinned memory for the h_PartialSums" << endl;
	}
	//Allocate global memory on the device
	if (!HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_A" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_A_Norm, MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_A" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_transpose, MatrixSizeInBytes))) {
		cout << "Cannot Allocate the memory for d_transpose" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_PartialSums, size_psum))) {
		cout << "Cannot Allocate the memory for d_PartialSums" << endl;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_PartialSums_stddev, size_psum))) {
		cout << "Cannot Allocate the memory for d_PartialSums" << endl;
	}
	//copy the matrix to the device
	if (!HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice))) {
		cout << "Cannot copy the matrix over" << endl;
	}

	//call the transpose kernel
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	cudaDeviceSynchronize();//sync the host and device
	float ElapsedTime{};
	//Create events to record the execution time for the partialsums kernel exec for means
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));
	//Declare 16 streams for execution
	cudaStream_t stream_1, stream_2, stream_3, stream_4;
	cudaStream_t stream_5, stream_6, stream_7, stream_8;
	cudaStream_t stream_9, stream_10, stream_11, stream_12;
	cudaStream_t stream_13, stream_14, stream_15, stream_16;
	//Create the 16 streams
	if (!HandleCUDAError(cudaStreamCreate(&stream_1))) {
		cout << "Cannot Create stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_2))) {
		cout << "Cannot Create stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_3))) {
		cout << "Cannot Create stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_4))) {
		cout << "Cannot Create stream_4" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_5))) {
		cout << "Cannot Create stream_5" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_6))) {
		cout << "Cannot Create stream_6" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_7))) {
		cout << "Cannot Create stream_7" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_8))) {
		cout << "Cannot Create stream_8" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_9))) {
		cout << "Cannot Create stream_9" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_10))) {
		cout << "Cannot Create stream_10" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_11))) {
		cout << "Cannot Create stream_11" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_12))) {
		cout << "Cannot Create stream_12" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_13))) {
		cout << "Cannot Create stream_13" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_14))) {
		cout << "Cannot Create stream_14" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_15))) {
		cout << "Cannot Create stream_15" << endl;
	}
	if (!HandleCUDAError(cudaStreamCreate(&stream_16))) {
		cout << "Cannot Create stream_16" << endl;
	}
	if (!HandleCUDAError(cudaEventRecord(start, 0))) {
		cout << "Unable to Record the start event" << endl;
	}
	for (unsigned int i = 0;i < nx;i += 16) {
		/* Within this loop, we are iterating through PartialReduction kernel launches
		We are going to allocate a row to each kernel launch, and in this for loop we are working with 16 rows at a time with 16 differnet kernel launches for Partial reduction
		this should be noted by the calls such as ((d_transpose + (i * ny)), ((d_transpose + ((i + 1) * ny)) etc. where we are pointing to the ith row and (i+1)th row respecitvely */
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_1 >> > ((d_transpose + (i * ny)), (d_PartialSums + (i * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_2 >> > ((d_transpose + ((i + 1) * ny)), (d_PartialSums + ((i + 1) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_3 >> > ((d_transpose + ((i + 2) * ny)), (d_PartialSums + ((i + 2) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_4 >> > ((d_transpose + ((i + 3) * ny)), (d_PartialSums + ((i + 3) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_5 >> > ((d_transpose + ((i + 4) * ny)), (d_PartialSums + ((i + 4) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_6 >> > ((d_transpose + ((i + 5) * ny)), (d_PartialSums + ((i + 5) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_7 >> > ((d_transpose + ((i + 6) * ny)), (d_PartialSums + ((i + 6) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_8 >> > ((d_transpose + ((i + 7) * ny)), (d_PartialSums + ((i + 7) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_9 >> > ((d_transpose + ((i + 8) * ny)), (d_PartialSums + ((i + 8) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_10 >> > ((d_transpose + ((i + 9) * ny)), (d_PartialSums + ((i + 9) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_11 >> > ((d_transpose + ((i + 10) * ny)), (d_PartialSums + ((i + 10) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_12 >> > ((d_transpose + ((i + 11) * ny)), (d_PartialSums + ((i + 11) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_13 >> > ((d_transpose + ((i + 12) * ny)), (d_PartialSums + ((i + 12) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_14 >> > ((d_transpose + ((i + 13) * ny)), (d_PartialSums + ((i + 13) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_15 >> > ((d_transpose + ((i + 14) * ny)), (d_PartialSums + ((i + 14) * blocks_per_grid)), ny);
		PartialReduction << <blocks_per_grid, threads_per_block, 0, stream_16 >> > ((d_transpose + ((i + 15) * ny)), (d_PartialSums + ((i + 15) * blocks_per_grid)), ny);
		/* Here, we are asynchronously copying the results back to the ith, (i+1)th etc row of d_PartialSums to the same row of h_PartialSums, a section of pinned*/
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + (i * blocks_per_grid)), (d_PartialSums + (i * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_1));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 1) * blocks_per_grid)), (d_PartialSums + ((i + 1) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_2));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 2) * blocks_per_grid)), (d_PartialSums + ((i + 2) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_3));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 3) * blocks_per_grid)), (d_PartialSums + ((i + 3) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_4));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 4) * blocks_per_grid)), (d_PartialSums + ((i + 4) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_5));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 5) * blocks_per_grid)), (d_PartialSums + ((i + 5) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_6));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 6) * blocks_per_grid)), (d_PartialSums + ((i + 6) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_7));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 7) * blocks_per_grid)), (d_PartialSums + ((i + 7) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_8));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 8) * blocks_per_grid)), (d_PartialSums + ((i + 8) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_9));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 9) * blocks_per_grid)), (d_PartialSums + ((i + 9) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_10));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 10) * blocks_per_grid)), (d_PartialSums + ((i + 10) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_11));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 11) * blocks_per_grid)), (d_PartialSums + ((i + 11) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_12));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 12) * blocks_per_grid)), (d_PartialSums + ((i + 12) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_13));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 13) * blocks_per_grid)), (d_PartialSums + ((i + 13) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_14));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 14) * blocks_per_grid)), (d_PartialSums + ((i + 14) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_15));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums + ((i + 15) * blocks_per_grid)), (d_PartialSums + ((i + 15) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_16));

	}
	//We must ensure all of the streams are synchronized after we iterate through the loop
	if (!HandleCUDAError(cudaStreamSynchronize(stream_1))) {
		cout << "Unable to perform stream synch with stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_2))) {
		cout << "Unable to perform stream synch with stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_3))) {
		cout << "Unable to perform stream synch with stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_4))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_5))) {
		cout << "Unable to perform stream synch with stream_5" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_6))) {
		cout << "Unable to perform stream synch with stream_6" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_7))) {
		cout << "Unable to perform stream synch with stream_7" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_8))) {
		cout << "Unable to perform stream synch with stream_8" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_9))) {
		cout << "Unable to perform stream synch with stream_9" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_10))) {
		cout << "Unable to perform stream synch with stream_10" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_11))) {
		cout << "Unable to perform stream synch with stream_11" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_12))) {
		cout << "Unable to perform stream synch with stream_12" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_13))) {
		cout << "Unable to perform stream synch with stream_13" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_14))) {
		cout << "Unable to perform stream synch with stream_14" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_15))) {
		cout << "Unable to perform stream synch with stream_15" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_16))) {
		cout << "Unable to perform stream synch with stream_16" << endl;
	}
	//After the streams are synched, we can now record the stop event
	if (!HandleCUDAError(cudaEventRecord(stop, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//We need to synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop))) {
		cout << "Unable to sync the stop even" << endl;
	}
	//Store the execution time of the partial sums in ElapsedTime
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Total time for stream partial sums: " << ElapsedTime << " ms" << endl;
	float* vect_temp;
	for (int i = 0; i < nx;i++) {
		//Iterate through the rows of the partial sum vector
		float temp = 0;//utilize this variable to store the sum of a row
		vect_temp = (h_PartialSums + (i * blocks_per_grid));//point to the start of a row at each iteration
		for (int j = 0;j < blocks_per_grid;j++) {
			temp += *(vect_temp + j);//Sum the entries of a row
		}
		h_mean[i] = temp / ny;//divide by the rows to get the mean
	}
	//Verify and Display the reference and mean
	Verification("PR Mean", ref, h_mean, nx);
	Display("h_Mean", h_mean, nx);
	Display("ref", ref, nx);
	//Create new events for find the execution time for the standard deviation reduction calculations
	cudaEvent_t start_stddev, stop_stddev;
	HandleCUDAError(cudaEventCreate(&start_stddev));
	HandleCUDAError(cudaEventCreate(&stop_stddev));
	//Record the start of the standard deviation reduction calculations with streams
	if (!HandleCUDAError(cudaEventRecord(start_stddev, 0))) {
		cout << "Unable to Record the start event" << endl;
	}
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	cudaDeviceSynchronize();
	//Calculate the standard deviation now using some form of reduction
	for (unsigned int i = 0;i < nx;i += 16) {
		/* Within this loop, we are iterating through PartialReduction_STDDEB kernel launches
		We are going to allocate a row to each kernel launch, and in this for loop we are working with 16 rows at a time with 16 differnet kernel launches for Partial reduction
		this should be noted by the calls such as ((d_transpose + (i * ny)), ((d_transpose + ((i + 1) * ny)) etc. where we are pointing to the ith row and (i+1)th row respecitvely */
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_1 >> > ((d_transpose + (i * ny)), (d_PartialSums_stddev + (i * blocks_per_grid)), h_mean[i], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_2 >> > ((d_transpose + ((i + 1) * ny)), (d_PartialSums_stddev + ((i + 1) * blocks_per_grid)), h_mean[i + 1], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_3 >> > ((d_transpose + ((i + 2) * ny)), (d_PartialSums_stddev + ((i + 2) * blocks_per_grid)), h_mean[i + 2], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_4 >> > ((d_transpose + ((i + 3) * ny)), (d_PartialSums_stddev + ((i + 3) * blocks_per_grid)), h_mean[i + 3], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_5 >> > ((d_transpose + ((i + 4) * ny)), (d_PartialSums_stddev + ((i + 4) * blocks_per_grid)), h_mean[i + 4], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_6 >> > ((d_transpose + ((i + 5) * ny)), (d_PartialSums_stddev + ((i + 5) * blocks_per_grid)), h_mean[i + 5], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_7 >> > ((d_transpose + ((i + 6) * ny)), (d_PartialSums_stddev + ((i + 6) * blocks_per_grid)), h_mean[i + 6], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_8 >> > ((d_transpose + ((i + 7) * ny)), (d_PartialSums_stddev + ((i + 7) * blocks_per_grid)), h_mean[i + 7], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_9 >> > ((d_transpose + ((i + 8) * ny)), (d_PartialSums_stddev + ((i + 8) * blocks_per_grid)), h_mean[i + 8], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_10 >> > ((d_transpose + ((i + 9) * ny)), (d_PartialSums_stddev + ((i + 9) * blocks_per_grid)), h_mean[i + 9], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_11 >> > ((d_transpose + ((i + 10) * ny)), (d_PartialSums_stddev + ((i + 10) * blocks_per_grid)), h_mean[i + 10], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_12 >> > ((d_transpose + ((i + 11) * ny)), (d_PartialSums_stddev + ((i + 11) * blocks_per_grid)), h_mean[i + 11], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_13 >> > ((d_transpose + ((i + 12) * ny)), (d_PartialSums_stddev + ((i + 12) * blocks_per_grid)), h_mean[i + 12], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_14 >> > ((d_transpose + ((i + 13) * ny)), (d_PartialSums_stddev + ((i + 13) * blocks_per_grid)), h_mean[i + 13], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_15 >> > ((d_transpose + ((i + 14) * ny)), (d_PartialSums_stddev + ((i + 14) * blocks_per_grid)), h_mean[i + 14], ny);
		PartialReduction_STDDEV << <blocks_per_grid, threads_per_block, 0, stream_16 >> > ((d_transpose + ((i + 15) * ny)), (d_PartialSums_stddev + ((i + 15) * blocks_per_grid)), h_mean[i + 15], ny);
		/* Here, we are asynchronously copying the results back to the ith, (i+1)th etc row of d_PartialSums to the same row of h_PartialSums_stddev, a section of pinned*/
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + (i * blocks_per_grid)), (d_PartialSums_stddev + (i * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_1));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 1) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 1) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_2));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 2) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 2) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_3));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 3) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 3) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_4));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 4) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 4) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_5));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 5) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 5) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_6));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 6) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 6) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_7));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 7) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 7) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_8));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 8) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 8) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_9));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 9) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 9) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_10));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 10) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 10) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_11));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 11) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 11) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_12));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 12) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 12) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_13));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 13) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 13) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_14));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 14) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 14) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_15));
		HandleCUDAError(cudaMemcpyAsync((h_PartialSums_stddev + ((i + 15) * blocks_per_grid)), (d_PartialSums_stddev + ((i + 15) * blocks_per_grid)), blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost, stream_16));
	}
	//we must ensure all of the streams are synchronize  before we proceed
	if (!HandleCUDAError(cudaStreamSynchronize(stream_1))) {
		cout << "Unable to perform stream synch with stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_2))) {
		cout << "Unable to perform stream synch with stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_3))) {
		cout << "Unable to perform stream synch with stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_4))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_5))) {
		cout << "Unable to perform stream synch with stream_5" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_6))) {
		cout << "Unable to perform stream synch with stream_6" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_7))) {
		cout << "Unable to perform stream synch with stream_7" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_8))) {
		cout << "Unable to perform stream synch with stream_8" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_9))) {
		cout << "Unable to perform stream synch with stream_9" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_10))) {
		cout << "Unable to perform stream synch with stream_10" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_11))) {
		cout << "Unable to perform stream synch with stream_11" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_12))) {
		cout << "Unable to perform stream synch with stream_12" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_13))) {
		cout << "Unable to perform stream synch with stream_13" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_14))) {
		cout << "Unable to perform stream synch with stream_14" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_15))) {
		cout << "Unable to perform stream synch with stream_15" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_16))) {
		cout << "Unable to perform stream synch with stream_16" << endl;
	}
	//Once the streams are synchronized, we now must record the stop event
	if (!HandleCUDAError(cudaEventRecord(stop_stddev, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event for standard deviation
	if (!HandleCUDAError(cudaEventSynchronize(stop_stddev))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Calculate the Elapsed time for the standard deviation partial reduction
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start_stddev, stop_stddev))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Exec time for the reduction for standard deviation: " << ElapsedTime << " ms" << endl;
	float* vect_temp_stddev;
	for (int i = 0;i < nx;i++) {
		//Iterate through the rows of the partial sum for stddev
		float temp = 0;
		vect_temp_stddev = h_PartialSums_stddev + (i * blocks_per_grid);//point to the start of a new row every iteration
		for (int j = 0;j < blocks_per_grid;j++) {
			temp += *(vect_temp_stddev + j);//sum the elements in each row
		}
		h_stddev[i] = sqrtf(temp / (ny - 1));
	}
	//Verify and Display the resutls
	Verification("PR Stddev", ref_stddev, h_stddev, nx);
	Display("h_stddev", h_stddev, nx);
	Display("ref", ref_stddev, nx);
	cudaEvent_t start_norm, stop_norm;
	HandleCUDAError(cudaEventCreate(&start_norm));
	HandleCUDAError(cudaEventCreate(&stop_norm));
	if (!HandleCUDAError(cudaEventRecord(start_norm, 0))) {
		cout << "Unable to Record the start event" << endl;
	}
	Transpose << <grid, block >> > (d_A, d_transpose, ny, nx);
	cudaDeviceSynchronize();
	//Calculate the standard deviation now using some form of reduction
	for (unsigned int i = 0;i < nx;i += 16) {
		/* Within this loop, we are iterating through NormalizeMatrix kernel launches
		We are going to allocate a row to each kernel launch, and in this for loop we are working with 16 rows at a time with 16 differnet kernel launches for Normalization
		this should be noted by the calls such as ((d_transpose + (i * ny)), ((d_transpose + ((i + 1) * ny)) etc. where we are pointing to the ith row and (i+1)th row respecitvely */
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_1 >> > ((d_transpose + (i * ny)), (d_A_Norm + (i * ny)), h_mean[i], h_stddev[i], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_2 >> > ((d_transpose + ((i + 1) * ny)), (d_A_Norm + ((i + 1) * ny)), h_mean[i + 1], h_stddev[i + 1], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_3 >> > ((d_transpose + ((i + 2) * ny)), (d_A_Norm + ((i + 2) * ny)), h_mean[i + 2], h_stddev[i + 2], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_4 >> > ((d_transpose + ((i + 3) * ny)), (d_A_Norm + ((i + 3) * ny)), h_mean[i + 3], h_stddev[i + 3], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_5 >> > ((d_transpose + ((i + 4) * ny)), (d_A_Norm + ((i + 4) * ny)), h_mean[i + 4], h_stddev[i + 4], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_6 >> > ((d_transpose + ((i + 5) * ny)), (d_A_Norm + ((i + 5) * ny)), h_mean[i + 5], h_stddev[i + 5], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_7 >> > ((d_transpose + ((i + 6) * ny)), (d_A_Norm + ((i + 6) * ny)), h_mean[i + 6], h_stddev[i + 6], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_8 >> > ((d_transpose + ((i + 7) * ny)), (d_A_Norm + ((i + 7) * ny)), h_mean[i + 7], h_stddev[i + 7], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_9 >> > ((d_transpose + ((i + 8) * ny)), (d_A_Norm + ((i + 8) * ny)), h_mean[i + 8], h_stddev[i + 8], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_10 >> > ((d_transpose + ((i + 9) * ny)), (d_A_Norm + ((i + 9) * ny)), h_mean[i + 9], h_stddev[i + 9], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_11 >> > ((d_transpose + ((i + 10) * ny)), (d_A_Norm + ((i + 10) * ny)), h_mean[i + 10], h_stddev[i + 10], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_12 >> > ((d_transpose + ((i + 11) * ny)), (d_A_Norm + ((i + 11) * ny)), h_mean[i + 11], h_stddev[i + 11], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_13 >> > ((d_transpose + ((i + 12) * ny)), (d_A_Norm + ((i + 12) * ny)), h_mean[i + 12], h_stddev[i + 12], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_14 >> > ((d_transpose + ((i + 13) * ny)), (d_A_Norm + ((i + 13) * ny)), h_mean[i + 13], h_stddev[i + 13], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_15 >> > ((d_transpose + ((i + 14) * ny)), (d_A_Norm + ((i + 14) * ny)), h_mean[i + 14], h_stddev[i + 14], ny);
		NormalizeMatrix << <blocks_per_grid, threads_per_block, 0, stream_16 >> > ((d_transpose + ((i + 15) * ny)), (d_A_Norm + ((i + 15) * ny)), h_mean[i + 15], h_stddev[i + 15], ny);
	}
	//We must synchronize the streams before we proceed
	if (!HandleCUDAError(cudaStreamSynchronize(stream_1))) {
		cout << "Unable to perform stream synch with stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_2))) {
		cout << "Unable to perform stream synch with stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_3))) {
		cout << "Unable to perform stream synch with stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_4))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_5))) {
		cout << "Unable to perform stream synch with stream_5" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_6))) {
		cout << "Unable to perform stream synch with stream_6" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_7))) {
		cout << "Unable to perform stream synch with stream_7" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_8))) {
		cout << "Unable to perform stream synch with stream_8" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_9))) {
		cout << "Unable to perform stream synch with stream_9" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_10))) {
		cout << "Unable to perform stream synch with stream_10" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_11))) {
		cout << "Unable to perform stream synch with stream_11" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_12))) {
		cout << "Unable to perform stream synch with stream_12" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_13))) {
		cout << "Unable to perform stream synch with stream_13" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_14))) {
		cout << "Unable to perform stream synch with stream_14" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_15))) {
		cout << "Unable to perform stream synch with stream_15" << endl;
	}
	if (!HandleCUDAError(cudaStreamSynchronize(stream_16))) {
		cout << "Unable to perform stream synch with stream_16" << endl;
	}
	//Once streams are synced, we can now record the stop of the exec with the stop event
	if (!HandleCUDAError(cudaEventRecord(stop_norm, 0))) {
		cout << "Unable to perform event records for stop" << endl;
	}
	//Synchronize the stop event
	if (!HandleCUDAError(cudaEventSynchronize(stop_norm))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	//Calculate the elapsed time from Normalizing the matrix
	if (!HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start_norm, stop_norm))) {
		cout << "Unable to find elapsed time between events" << endl;
	}
	cout << "Exec time for the normalization of the input matrix: " << ElapsedTime << " ms" << endl;
	//Copy the Normalized matrix from the device to the host
	if (!HandleCUDAError(cudaMemcpy(h_A_Norm, d_A_Norm, MatrixSizeInBytes, cudaMemcpyDeviceToHost))) {
		cout << "Cannot copy the normalized matrix back to the host" << endl;
	}
	//Display and Verify the results of normalization
	Verification("Normalized Matrix", norm_ref, h_A_Norm, ny * nx);
	cout << "GPU Normalized Matrix" << endl;
	Display("h_A_Norm", h_A_Norm, ny * nx);
	cout << "CPU Normalized Matrix" << endl;
	Display("norm_ref", norm_ref, ny * nx);
	cout << "Exec time for the reduction for Normalization: " << ElapsedTime << " ms" << endl;
	//Free the pinned memory 
	if (!HandleCUDAError(cudaFreeHost(h_PartialSums))) {
		cout << "Unable to free the h_PartialSums memory on the host" << endl;
	}
	if (!HandleCUDAError(cudaFreeHost(h_PartialSums_stddev))) {
		cout << "Unable to free the h_PartialSums memory on the host" << endl;
	}
	//destory the streams we utilized 
	if (!HandleCUDAError(cudaStreamDestroy(stream_1))) {
		cout << "Unable to perform stream synch with stream_1" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_2))) {
		cout << "Unable to perform stream synch with stream_2" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_3))) {
		cout << "Unable to perform stream synch with stream_3" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_4))) {
		cout << "Unable to perform stream synch with stream_4" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_5))) {
		cout << "Unable to perform stream synch with stream_5" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_6))) {
		cout << "Unable to perform stream synch with stream_6" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_7))) {
		cout << "Unable to perform stream synch with stream_7" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_8))) {
		cout << "Unable to perform stream synch with stream_8" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_9))) {
		cout << "Unable to perform stream synch with stream_9" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_10))) {
		cout << "Unable to perform stream synch with stream_10" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_11))) {
		cout << "Unable to perform stream synch with stream_11" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_12))) {
		cout << "Unable to perform stream synch with stream_12" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_13))) {
		cout << "Unable to perform stream synch with stream_13" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_14))) {
		cout << "Unable to perform stream synch with stream_14" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_15))) {
		cout << "Unable to perform stream synch with stream_15" << endl;
	}
	if (!HandleCUDAError(cudaStreamDestroy(stream_16))) {
		cout << "Unable to perform stream synch with stream_16" << endl;
	}
	//Destroy all of the events we have created
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));
	HandleCUDAError(cudaEventDestroy(start_stddev));
	HandleCUDAError(cudaEventDestroy(stop_stddev));
	HandleCUDAError(cudaEventDestroy(start_norm));
	HandleCUDAError(cudaEventDestroy(stop_norm));

	//Free all of the global memory on the device
	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_A_Norm));
	HandleCUDAError(cudaFree(d_PartialSums));
	HandleCUDAError(cudaFree(d_PartialSums_stddev));
	HandleCUDAError(cudaFree(d_transpose));
	HandleCUDAError(cudaDeviceReset());
}
