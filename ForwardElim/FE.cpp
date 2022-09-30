#include "FE.h"


int main() {
	float* A{};
	float* b{};
	int N = 3;
	int Size_A = N * N;
	int Size_b = N;
	A = new float[Size_A];
	b = new float[N];
	Random(A, N, N);
	Random(b, 1, N);

	cout << "A:" << endl;
	DisplayMatrix(A, N, Size_A);
	cout << "b:" << endl;
	DisplayMatrix(b, N, Size_b);
	Diag_Dominant(A, N);
	cout << "Diag Dominant A:" << endl;
	DisplayMatrix(A, N, Size_A);
	FE(A, b, N);
	cout << "FE Matrix: " << endl;
	DisplayMatrix(A, N, Size_A);
	cout << "b:" << endl;
	DisplayMatrix(b, N, Size_b);
	return 0;
}