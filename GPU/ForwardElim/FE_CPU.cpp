#include "FE.h"


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