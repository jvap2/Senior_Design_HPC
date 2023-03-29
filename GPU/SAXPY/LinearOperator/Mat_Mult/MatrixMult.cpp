#include "MatrixMult.h"
#include "GPUErrors.h"

int main()
{
	srand((unsigned)time(NULL));

	int rows = 4;//This is left shifting to get 2^10, and we want this to be to power of 2 for GPU
	int cols = 4;//256x256 matrix if both are 1<<8
	cout << "Matrix Multiplication of Size: " << rows << "x" << cols << endl;
	float* A, * B, * C;//Defining three matrices
	int C_size=(rows*(rows+1))/2;
	A = new float[rows];
	B = new float[rows];
	C = new float[C_size];

	Generate_Vector(A,cols);
	Generate_Vector(B,cols);


	//Host Multiplication
	cpuMatrixMult(A, B, C, rows, cols);
	int count=0;
	cout<<"C"<<endl;
	for(int i=0; i<C_size;i++){
		cout<<C[i]<<endl;
	}
	cout<<"A"<<endl;
	for(int i=0;i<rows;i++){
		cout<<A[i]<<endl;
	}
	cout<<"B"<<endl;
	for(int j=0; j<rows;j++){
		cout<<B[j]<<endl;
	}

	// DisplayMatrix("A", A, rows, cols);
	// DisplayMatrix("B", B, rows, cols);
	// DisplayMatrix("C", C, rows, cols);

	// float* gpuC;
	// gpuC = new float[rows * cols];

	// gpuMultHelper(A, B, gpuC, C, rows, cols);

	delete[] A;
	delete[] B;
	delete[] C;
	// delete[] gpuC;

	return 0;
}