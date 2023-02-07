#include "FE.h"


void Random(float* Mat, int row, int col) {
	//Intended for row major, will make horizontal vector
	float* p = Mat;
	for (int i{}; i < row;i++) {
		for (int j{}; j < col;j++) {
			p[j] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		p += col;
	}
}

void Diag_Dominant(float* Mat, int N) {
	int max_loc{};
	float max{};
	float temp{};
	for (int i{}; i < N;i++) {
		max = abs(Mat[i * N]);
		for (int j{ 1 }; j < N;j++) {
			if (abs(Mat[i * N + j]) > max) {
				max_loc = j;
				max = Mat[i * N + j];
			}
		}
		temp = Mat[i * N + i];
		Mat[i * N + i] = max;
		Mat[i * N + max_loc] = temp;
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