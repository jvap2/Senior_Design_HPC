#include "CG_LinOp.h"


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
			fSum += (A_T[abs(i-j)] * b[j]);
		}
        b_new[i]=fSum;
	}
}


void Generate_Vector(float* in, int size){
    for(int i{}; i<size;i++){
        in[i]=(float)(rand())/(float)(RAND_MAX);
    }
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

void Verify(float* iter, float* res, int size, bool* check, int* count){
	for(int i{}; i<size; i++){
		float dif=fabsf(*(iter+i)-*(res+i));
		if(dif>1e-3){
			*check=false;
			*count=i;
			return;
		}
	}
}


float L_2(float* in, int size){
    float val{};
    for(int i{}; i<size;i++){
        val+=powf(in[i],2.0f);
    }
    val=sqrtf(val);
    return val;
}


void C_G(float* A, float* r, float* r_old, float* d, float* d_old, float* x, float* x_old, float* beta, float* lamdba, int size, int* iter){
    float Ad[size]={};
    float lamd_d[size]={};
    float beta_d[size]={};
    float lambd_AD[size]={};
    float temp_1{};
    float temp_2{};
    int MaxIter=10*size;
    float norm{};
    while(*(iter)<MaxIter){
        cpuMatrixVect(A,d_old,Ad,size,size);
        temp_1=Dot_Product(r_old,r_old,size);
        temp_2=Dot_Product(d_old,Ad,size);
        if(fabsf(temp_2)<1e-8){
            return;
        }
        (*lamdba)=temp_1/temp_2;
        float neg_lamb=-(*lamdba);
        Const_Vect_Mult(d_old,lamd_d,*lamdba,size);
        Const_Vect_Mult(Ad,lambd_AD,neg_lamb,size);
        cpuVectorAddition(x_old,lamd_d,x,size);
        cpuVectorAddition(r_old,lambd_AD,r,size);
        norm=L_2(r,size);
        if(norm<1e-6){
            return;
        }
        temp_2=Dot_Product(r,r,size);
        *beta=temp_2/temp_1;
        Const_Vect_Mult(d_old,beta_d,*beta,size);
        cpuVectorAddition(r,beta_d,d,size);
        for(int i=0;i<size;i++){
            d_old[i]=d[i];
            r_old[i]=r[i];
            x_old[i]=x[i];

        }
        *(iter)+=1;
    }
}



void Diag_Dominant_Opt(float* Mat, int n) {
	float max_row{};
	for (int i{}; i < n;i++) {
		max_row+=fabsf(Mat[i]);
	}
	Mat[0]=max_row;
}
