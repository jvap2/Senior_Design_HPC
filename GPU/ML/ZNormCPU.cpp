#include "ZNorm.h"

void cpuMean(float* A, float* mean, const int ny, const int nx)
{
	float fSum;
	chrono::time_point<high_resolution_clock> start, end;
	start = high_resolution_clock::now();
	for (int i = 0;i < nx;i++) {
		//Iterate through the columns with the outermost loop
		fSum = 0;//set fSum to 0 for each column to calculate new sum for each column for the mean
		for (int j = 0;j < ny;j++) {
			//iterate through the rows of the ith column collecting the sum of the elements
			fSum += A[j * nx + i];
		}
		mean[i] = fSum / ny;//get the ith sum by dividing fSum, the sum of the ith column by the number of rows
	}
	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	double computeTime{};
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "CPU Mean Computation time: " << computeTime << " usecs" << endl;
}

void cpuStdDev(float* A, float* mean, float* stddev, const int ny, const int nx)
{
	float fSumSquare;
	chrono::time_point<high_resolution_clock> start, end;
	start = high_resolution_clock::now();
	for (int i = 0;i < nx;i++) {
		//Iterate through each column
		fSumSquare = 0;//Set fSumSquare to 0 for each iteration
		float mu = mean[i];//fetch column i's mean for the sample mean standard deviation calculation
		for (int j = 0;j < ny;j++) {
			fSumSquare += powf(A[j * nx + i]-mu,2.0f);//Calculate the summation portion of the standard deviation equation
		}
		stddev[i] = sqrtf(fSumSquare / (ny-1));//Complete the sdev equation
	}
	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	double computeTime{};
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "CPU Standard Deviation Computation time: " << computeTime << " usecs" << endl;
}

void cpuNormalizeMat(float* A, float* A_norm, float* mean, float* stddev, const int ny, const int nx) {
	chrono::time_point<high_resolution_clock> start, end;
	start = high_resolution_clock::now();
	for (int i = 0; i < nx;i++) {
		//Iterate through the columns, and fetch the ith columns mean and stddev
		float mu = mean[i];
		float sdev = stddev[i];
		for (int j = 0;j < ny;j++) {
			A_norm[j * nx + i] = (A[j * nx + i] - mu) / sdev;// Normalize each value in a column
		}
	}
	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	double computeTime{};
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "CPU Normalization Computation time: " << computeTime << " usecs" << endl;
}

