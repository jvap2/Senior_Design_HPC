#include "SVD.h"

void InitializeMatrix(float *matrix, int ny, int nx)
{
	float *p = matrix;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = ((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		p += nx;
	}
}

void CopyMat(float *A, float*B, int ny, int nx){

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			B[j] = A[j];
		}
		A += nx;
		B += nx;
	}
}

void ZeroMatrix(float *temp, const int ny, const int nx)
{
	float *p = temp;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = 0.0f;
		}
		p += nx;
	}
}

void ZeroVector(float *temp, const int size)
{
	float *p = temp;
	for (int j = 0; j < size; j++)
	{
		p[j] = 0.0f;
	}
}

void eye_1(float* temp, int size){
	ZeroVector(temp,size);
	temp[0]=1;
}


void DisplayMatrix(string name, float* temp, const int ny, const int nx)
{
	if (ny < 6 && nx < 6)
	{
		cout << name << endl;
		for (int i = 0; i < ny; i++)
		{
			for (int j = 0; j < nx; j++)
			{
				cout << setprecision(6) << temp[(i * nx) + j] << "\t";
			}
			cout << endl;
		}
	}
}

void SVDVerification(float* hostC, float* gpuC, const int ny, const int nx)
{
	float fTolerance = 1.0E-01;
	float* p = hostC;
	float* q = gpuC;
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			if (fabs(fabsf(p[j]) - fabsf(q[j])) > fTolerance)
			{
				cout << "Error" << endl;
				cout << "\thostC[" << (i + 1) << "][" << (j + 1) << "] = " << p[j] << endl;
				cout << "\tgpuC[" << (i + 1) << "][" << (j + 1) << "] = " << q[j] << endl;
				return;
			}
		}
		p += nx;
		q += nx;
	}
}

void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx)
{
	for (int y = 0; y < ny; y++)
	{
		for (int x = 0; x < nx; x++)
		{
			//Load Coalesced and Store stride
			matrixTranspose[x * ny + y] = matrix[y * nx + x];
		}
	}
}

void IdentityMatrix(float* temp, int size){
	ZeroMatrix(temp,size,size);
	for(int i{}; i<size; i++){
		temp[i*size+i]=1.0f;
	}
}