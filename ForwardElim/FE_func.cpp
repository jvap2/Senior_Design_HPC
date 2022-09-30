#include "FE.h"


void Random(float* Mat, int row, int col) {
	//Intended for row major, will make horizontal vector
	for (int i{}; i < row;i++) {
		for (int j{}; j < col;j++) {
			Mat[j] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		Mat += col;
	}
}


void FE(float* A, float* b, int N) {
	float ratio{};
	for (int k{}; k < N;k++) {
		for (int i{ k + 1 };i < N;i++) {
			ratio = A[i * N + k] / A[k * N + k];
			for (int j{ k }; j < N;j++) {
				A[i * N + j] -= ratio * A[k * N + j];
			}
			b[i] -= ratio * b[k];
		}
	}
}

void DisplayMatrix(float* Mat, int N, int Size) {
	int count{};
	if (N < 10) {
		for (int i{}; i < Size; i++) {
			cout << Mat[i]<<'\t';
			count++;
			if (count == (N)){
				count = 0;
				cout << endl;
			}
		}
	}
	else {
		for (int i{};i < 10;i++) {
			cout << Mat[i] << '\t';
		}
		cout << endl;
		cout << "........." << endl;
		for (int i{ Size - 10 };i < Size;i++) {
			cout << Mat[i] << '\t';
		}
		cout << endl;
	}
	cout << endl;
}