#include "MatrixMult.h"
#include "GPUErrors.h"

int main()
{
	srand((unsigned)time(NULL));

	int rows = 1<<12;//This is left shifting to get 2^10, and we want this to be to power of 2 for GPU
	int cols = 1<<12;//256x256 matrix if both are 1<<8
	cout << "Matrix Multiplication of Size: " << rows << "x" << cols << endl;
	float* A, * B, * C;//Defining three matrices
	int C_size=(rows*cols);
	cout<<C_size<<endl;
	A = new float[rows];
	B = new float[rows];
	C = new float[C_size];
	Generate_Vector(A,cols);
	Generate_Vector(B,cols);

	//Host Multiplication
	cpuMatrixMult(A, B, C, rows, cols);
	// DisplayMatrix("A", A, rows, cols);
	// DisplayMatrix("B", B, rows, cols);
	// DisplayMatrix("C", C, rows, cols);

	float* gpuC;
	gpuC = new float[C_size];

	gpuMultHelper(A, B, gpuC, C, rows, cols);

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] gpuC;

	return 0;
}