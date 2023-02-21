#include "CG.hpp"


void TransposeOnCPU(int* matrix, int* matrixTranspose, int ny, int nx)
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
void cpuMatrixMult(int* A, int* A_T, int* C, const int ny, const int nx)
{
	int fSum;
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			fSum = 0.0f;
			for (int k = 0; k < nx; k++)
			{
				fSum += (A[(i * nx) + k] * A_T[(k * nx) + j]);
			}
			C[(i * nx) + j] = fSum;
		}
	}
}
void cpuMatrixVect(int* A_T, int* b, int* b_new, const int ny, const int nx)
{
	int fSum;
	for (int i = 0; i < ny; i++)
	{
        fSum = 0.0f;
		for (int j = 0; j < nx; j++)
		{
			fSum += (A_T[(i * nx) + j] * b[j]);
		}
        b_new[i]=fSum;
	}
}

void Diag_Dominant_Opt(int* Mat, int N) {
	int max_row{};
	int max_col{};
	for (int i{}; i < N;i++) {
		for (int j{ 1 }; j < N;j++) {
		    max_row+=fabsf(Mat[i*N+j]);
		    max_col+=fabsf(Mat[j*N+i]);
		}
		if(max_col>=max_row){
		    Mat[i*N+i]=max_col;
		}
		else{
		    Mat[i * N + i] = max_row;
		}
		max_row=0;
		max_col=0;
	}
}


void Generate_Vector(int* in, int size){
    for(int i{}; i<size;i++){
        in[i]=(int)(rand())/(int)(RAND_MAX);
    }
}


void InitializeMatrix(int *matrix, int ny, int nx)
{
	int *p = matrix;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = (int)(rand())/(int)(RAND_MAX);
		}
		p += nx;
	}
}

int Dot_Product(int* in_1,int* in_2, int size){
    int res{};
    for(int i{}; i<size; i++){
        res+=in_1[i]*in_2[i];
    }
    return res;
}

void cpuVectorAddition(int* A, int* B, int* C, int size){
    for(int i=0; i<size; i++){
        C[i]=A[i]+B[i];
    }
}

void Const_Vect_Mult(int* vect, int* out, int scalar, int size){
	for(int i=0; i<size; i++){
		out[i]=scalar*vect[i];
	}
}

void vector_subtract(int* in_1, int* in_2, int* out, int size){
	for(int i{}; i<size;i++){
		out[i]=in_1[i]-in_2[i];
	}
}

void Verify(int* iter, int* res, int size){
	for(int i{}; i<size; i++){
		int dif=fabsf(*(iter+i)-*(res+i));
		if(dif>1e-4){
			std::cout<<"GPU["<<i<<"]="<<*(iter+i)<<std::endl;
			std::cout<<"CPU["<<i<<"]="<<*(res+i)<<std::endl;
			std::cout<<"Error with the Dot Product"<<std::endl;
			return;
		}
	}
}


int L_2(int* in, int size){
    int val{};
    for(int i{}; i<size;i++){
        val+=powf(in[i],2.0f);
    }
    val=sqrtf(val);
    return val;
}


void C_G(int* A, int* r, int* r_old, int* d, int* d_old, int* x, int* x_old, int beta, int lamdba, int size, int iter[1]){
    int Ad[size]={};
    int lamd_d[size]={};
    int beta_d[size]={};
    int lambd_AD[size]={};
    int temp_1{};
    int temp_2{};
    int MaxIter=10*size;
    int norm{};
    int count=0;
    while(count<MaxIter){
        cpuMatrixVect(A,d_old,Ad,size,size);
        temp_1=Dot_Product(r_old,r_old,size);
        temp_2=Dot_Product(d_old,Ad,size);
        if(fabsf(temp_2)<1e-8){
            return;
        }
        lamdba=temp_1/temp_2;
        int neg_lamb=-lamdba;
        Const_Vect_Mult(d_old,lamd_d,lamdba,size);
        Const_Vect_Mult(Ad,lambd_AD,neg_lamb,size);
        cpuVectorAddition(x_old,lamd_d,x,size);
        cpuVectorAddition(r_old,lambd_AD,r,size);
        norm=L_2(r,size);
        if(norm<1e-6){
            return;
        }
        temp_2=Dot_Product(r,r,size);
        beta=temp_2/temp_1;
        Const_Vect_Mult(d_old,beta_d,beta,size);
        cpuVectorAddition(r,beta_d,d,size);
        for(int i=0;i<size;i++){
            d_old[i]=d[i];
            r_old[i]=r[i];
            x_old[i]=x[i];

        }
        count+=1;
    }
    iter[0]=count;
}
