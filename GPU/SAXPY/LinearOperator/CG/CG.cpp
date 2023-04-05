#include "CG.h"
#include "GPUErrors.h"


int main(){
    float *A,*A_T,*A_res, *b, *b_res, *r, *r_old, *d, *d_old, *x, *x_old, *Ax, *b_check;
    float *r_GPU,*r_old_GPU,*d_GPU,*d_old_GPU,*x_GPU,*x_old_GPU;
    float lambda{}, beta{};
    float lambda_GPU{}, beta_GPU{};
    int ny=2048;
    int nx=2048;
    if(ny!=nx){
        return -1;
    }
    cout<<"Size="<<ny<<endl;
    Ax=new float[ny];
    A=new float[nx];
    b=new float[ny];
    b_res=new float[ny];
    r=new float[ny];
    r_old=new float[ny];
    d=new float[ny];
    d_old=new float[ny];
    x=new float[ny];
    x_old=new float[ny];
    r_GPU=new float[ny];
    r_old_GPU=new float[ny];
    d_GPU=new float[ny];
    d_old_GPU=new float[ny];
    x_GPU=new float[ny];
    x_old_GPU=new float[ny];
    b_check=new float[ny];
    int iter{};
    Generate_Vector(A,nx);
    Diag_Dominant_Opt(A,nx);
    Generate_Vector(b,ny);
    Generate_Vector(x_old,ny);
    std::cout<<"x"<<endl;
    cpuMatrixVect(A, x_old, Ax, ny, nx);
    vector_subtract(b,Ax,r_old,ny);
    vector_subtract(b,Ax,r_old_GPU,ny);

    for(int i=0; i<ny;i++){
        x_old_GPU[i]=x_old[i];
        d_old[i]=r_old[i];
        d_old_GPU[i]=d_old[i];
    }
    chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    C_G(A,r,r_old,d,d_old,x,x_old,beta,lambda,ny,&iter);
    end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
    CG_Helper(A,x,r_GPU,r_old_GPU,d_GPU,d_old_GPU,x_GPU,x_old_GPU,beta_GPU,lambda_GPU,ny,iter);
    Display("CPU",x,ny);
    delete[] Ax;
    delete[] A;
    delete[] b;
    delete[] b_res;
    delete[] r;
    delete[] r_old;
    delete[] d;
    delete[] d_old;
    delete[] x;
    delete[] x_old;
    delete[] r_GPU;
    delete[] r_old_GPU;
    delete[] d_GPU;
    delete[] d_old_GPU;
    delete[] x_GPU;
    delete[] x_old_GPU;

    return 0;

}