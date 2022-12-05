#include "ZNorm.h"

void InitializeMatrix(float* matrix, int ny, int nx)
{
	const int range_from = 0.0;
	const int range_to = 5.0;
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_real_distribution<float>  distr(range_from, range_to);
	float *p = matrix;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			//p[j] = 1.0f;
			p[j] = distr(generator);
		}
		p += nx;
	}
}

void DisplayMatrix(float* temp, const int ny, const int nx)
{
	if (ny < 6 && nx < 6)
	{
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
//Function to display the mean and standard deviation
void Display(string name, float* temp, const int nx)
{
	cout << name << endl;
	if(nx>10)
	{ 
		for (int j = 0; j < 5; j++)
		{
			cout << temp[j] << ", ";
		}
		cout << ",. . .,";
		for (int j = (nx-5); j < nx; j++)
		{
			cout << temp[j] << ", ";
		}
	}
	else {
		for (int j = 0; j < nx; j++)
		{
			cout << temp[j] << ", ";
		}
	}
	cout << endl;
}

void Verification(string name, float* hostC, float* gpuC, const int nx)
{
	float fTolerance = 1.0E-03;
	float* p = hostC;
	float* q = gpuC;
	for (int j = 0; j < nx; j++)
	{
		if (fabs(p[j] - q[j]) > fTolerance)
		{
			cout << name << endl;
			cout << "Error" << endl;
			cout << "\thostC[" << (j + 1) << "] = " << p[j] << endl;
			cout << "\tgpuC[" << (j + 1) << "] = " << q[j] << endl;
			return;
		}
	}
}

void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx)
{
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int y = 0; y < ny; y++)
	{
		for (int x = 0; x < nx; x++)
		{
			//Load Coalesced and Store stride
			matrixTranspose[x * ny + y] = matrix[y * nx + x];
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Transpose Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}