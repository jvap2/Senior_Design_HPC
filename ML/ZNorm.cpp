//Tiled Matrix Multiplication
#include "ZNorm.h"
#include "GPUErrors.h"

int main()
{
	srand((unsigned)time(NULL));

	int rows = 16384;
	int cols = 4096;
	cout << "Matrix Size: " << rows << "x" << cols << endl;
	
	//Define Data Matrix
	float* A;
	A = new float[rows * cols];

	//Define a vector to store the mean of the columns of the Data Matrix
	float* MeanVector_cpu;
	MeanVector_cpu = new float[cols];

	//Define a vector to store the standard deviation of the coloumns of the Data Matrix
	float* StdDevVector_cpu;
	StdDevVector_cpu = new float[cols];

	//Define Transpose Matrix
	float* AT;
	AT = new float[rows * cols];

	//Define Normalized Matrix
	float* A_norm_cpu;
	A_norm_cpu = new float[rows * cols];
	float* A_norm_gpu;
	A_norm_gpu = new float[rows * cols];
	float* A_norm_cpu_T;
	A_norm_cpu_T = new float[rows * cols];
	//GPU Mean Vector
	float* MeanVector_gpu;
	MeanVector_gpu = new float[cols];

	//Define a vector to store the standard deviation of the coloumns of the Data Matrix
	float* StdDevVector_gpu;
	StdDevVector_gpu = new float[cols];

	InitializeMatrix(A, rows, cols);
	DisplayMatrix(A, rows, cols);

	//Host Matrix Column Mean Computation
	cpuMean(A,MeanVector_cpu,rows,cols);
	Display("Matrix Column Mean (CPU):", MeanVector_cpu, cols);

	//Host Matrix Column Standard Deviation Computation
	cpuStdDev(A, MeanVector_cpu, StdDevVector_cpu, rows, cols);
	Display("Matrix Column Standard Deviation (CPU: ", StdDevVector_cpu, cols);
	/*cpuNormalizeMat(A, A_norm, MeanVector_cpu, StdDevVector_cpu, rows, cols);*/
	//GPU Matrix Mean Naive Computation
	cpuNormalizeMat(A, A_norm_cpu, MeanVector_cpu, StdDevVector_cpu, rows, cols);
	cpuNormalizeMat(A, A_norm_cpu_T, MeanVector_cpu, StdDevVector_cpu, rows, cols);
	cout << endl<< "Computing Column Means with a Thread per Column " << endl;
	cout << "Naive Implementation:" << endl;
	cout << "--------------------------------------------------------------------" << endl;
	gpuThreadHelper(A, A_norm_gpu, MeanVector_gpu, MeanVector_cpu, StdDevVector_gpu, StdDevVector_cpu,A_norm_cpu,rows, cols);
	cout << "--------------------------------------------------------------------" << endl;

	TransposeOnCPU(A, AT, rows, cols);
	TransposeOnCPU(A_norm_cpu, A_norm_cpu_T, rows, cols);
	//////GPU Matrix Mean with Parallel Reduction using the transpose of the input matrix
	cout <<endl<< "Computing Column Means using GPU Parallel Reduction on each column, 4 Streams" << endl;
	cout << "--------------------------------------------------------------------" << endl;
	gpuPRMeanHelper(A,A_norm_gpu, MeanVector_gpu, MeanVector_cpu, StdDevVector_gpu, StdDevVector_cpu, A_norm_cpu_T, rows, cols);
	cout << endl << "Computing Column Means using GPU Parallel Reduction on each column, Default Stream" << endl;
	cout << "--------------------------------------------------------------------" << endl;
	gpuPRMeanHelper_SingleStream(A, A_norm_gpu, MeanVector_gpu, MeanVector_cpu, StdDevVector_gpu, StdDevVector_cpu, A_norm_cpu_T, rows, cols);
	cout << "--------------------------------------------------------------------" << endl;
	cout << endl << "Computing Column Means using GPU Parallel Reduction on each column, 16 Streams" << endl;
	cout << "--------------------------------------------------------------------" << endl;
	gpuPRMeanHelper_16Stream(A, A_norm_gpu, MeanVector_gpu, MeanVector_cpu, StdDevVector_gpu, StdDevVector_cpu, A_norm_cpu_T, rows, cols);



	delete[] A;
	delete[] AT;
	delete[] A_norm_gpu;
	delete[] A_norm_cpu;
	delete[] A_norm_cpu_T;
	delete[] MeanVector_cpu;
	delete[] MeanVector_gpu;
	delete[] StdDevVector_cpu;
	delete[] StdDevVector_gpu;

	return 0;
}