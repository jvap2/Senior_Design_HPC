#include "CG.h"




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

void cpuMatrixVect(float* A_T, float* b, float* b_new, const int ny, const int nx)
{
	float fSum;
	for (int i = 0; i < ny; i++)
	{
        fSum = 0.0f;
		for (int j = 0; j < nx; j++)
		{
			if(j>i){
				fSum += (A_T[j-i] * b[j]);
			}
			else if(i>j){
				fSum += (A_T[i-j] * b[j]);
			}
			else{
				fSum+=(A_T[0]*b[j]);
			}
		}
        b_new[i]=fSum;
	}
}

void Diag_Dominant_Opt(float* Mat, int N) {
	float max_row{};
	for (int i{}; i < N;i++) {
		max_row+=fabsf(Mat[i]);
	}
	Mat[0]=max_row;
}




void InitializeMatrix(float *matrix, int ny, int nx)
{
	float *p = matrix;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = (float)(rand())/(float)(RAND_MAX);
		}
		p += nx;
	}
}

float Dot_Product(float* in_1,float* in_2, int size){
    float res{};
    for(int i{}; i<size; i++){
        res+=in_1[i]*in_2[i];
    }
    return res;
}

void cpuVectorAddition(float* A, float* B, float* C, int size){
    for(int i=0; i<size; i++){
        C[i]=A[i]+B[i];
    }
}

void Const_Vect_Mult(float* vect, float* out, float scalar, int size){
	for(int i=0; i<size; i++){
		out[i]=scalar*vect[i];
	}
}

void vector_subtract(float* in_1, float* in_2, float* out, int size){
	for(int i{}; i<size;i++){
		out[i]=in_1[i]-in_2[i];
	}
}

void Verify(float* iter, float* res, int size){
	for(int i{}; i<size; i++){
		float dif=fabsf(*(iter+i)-*(res+i));
		if(dif>1e-4){
			cout<<"GPU["<<i<<"]="<<*(iter+i)<<endl;
			cout<<"CPU["<<i<<"]="<<*(res+i)<<endl;
			cout<<"Error with the Dot Product"<<endl;
			return;
		}
	}
}

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

float L_2(float* in, int size){
    float val{};
    for(int i{}; i<size;i++){
        val+=powf(in[i],2.0f);
    }
    val=sqrtf(val);
    return val;
}

void Generate_Vector(float* in, int size){
    for(int i{}; i<size;i++){
        in[i]=(float)(rand())/(float)(RAND_MAX);
    }
}