// Includes
#include <hls_vector.h>
#include <hls_stream.h>
#include "assert.h"
#include <iostream>
#include <cmath>


#define N 64
#define nx 64
#define ny 64
void dot_p(int a[N], int b[N], int out[1])
{
  int a_int[N],b_int[N];
#pragma HLS array_partition  variable=a_int dim=1 complete
#pragma HLS array_partition  variable=b_int dim=1 complete
  int product = 0;

  for(int i=0;i<N;i++) {
  #pragma HLS pipeline
    a_int[i] = a[i];
  }
  for(int i=0;i<N;i++) {
  #pragma HLS pipeline
    b_int[i] = b[i];
  }

  for(int i=0;i<N;i++) {
  #pragma HLS unroll
    product += a_int[i] * b_int[i];
  }
 out[0] = product;

}

void vadd_p(int a[N], int b[N], int out[N]){
	  int a_int[N],b_int[N];
	#pragma HLS array_partition  variable=a_int dim=1 complete
	#pragma HLS array_partition  variable=b_int dim=1 complete
	  int product = 0;

	  copy_a:for(int i=0;i<N;i++) {
	  #pragma HLS pipeline
	    a_int[i] = a[i];
	  }
	  copy_b: for(int i=0;i<N;i++) {
	  #pragma HLS pipeline
	    b_int[i] = b[i];
	  }

	  final: for(int i=0;i<N;i++) {
	  #pragma HLS unroll
	    out[i] = a_int[i] + b_int[i];
	  }
}

void MatVec_Mult(int A[N*N], int b[N], int b_new[N]){
#pragma HLS array partition variable=A complete
#pragma HLS array partition variable=b In complete
	int fSum;
	row: for (int i = 0; i < ny; i++)
	{
#pragma HLS pipeline II=1
        fSum = 0.0f;
		col: for (int j = 0; j < nx; j++)
		{
			fSum += (A[(i * nx) + j] * b[j]);
		}
        b_new[i]=fSum;
	}
}

void Copy(int new_v[N], int old_v[N]){
	cop: for(int i=0; i<N; i++){
#pragma HLS unroll
		new_v[i]=old_v[i];
	}
}

void const_Vect_mult(int scalar[1], int vect[N], int out[N]){
	mult: for (int i = 0; i< N; i++){
#pragma HLS unroll
		out[i]=(scalar[0])*(vect[i]);
	}
}

void compt(int scal_1, int scal_2, int out[1],int flag){
	if(flag==1){
		out[0]=scal_1/scal_2;
	}
	else{
		out[0]=-scal_1/scal_2;
	}

}
void final(int A[N*N],
		int r[N],
		int r_old[N],
		int d[N],
		int d_old[N],
		int x[N],
		int x_old[N],
		int beta[1],
		int lambda[1],
		int iter[1]){
	int Ad[N]{};
	int r_dot{};
	int r_old_dot{};
	int d_Ad_dot{};
	int mult_d[N]{};
	int neg_lamb[1]{};
	int count = *iter;
	for(int i{}; i<count;i++){
		MatVec_Mult(A, d_old, Ad);
		dot_p(r_old,r_old, &r_old_dot);
		dot_p(d_old,Ad,&d_Ad_dot);
		compt(r_old_dot,d_Ad_dot,lambda,1);
		compt(r_old_dot,d_Ad_dot,neg_lamb,0);
		const_Vect_mult(lambda, d_old,mult_d);
		const_Vect_mult(neg_lamb, Ad,Ad);
		vadd_p(mult_d,x_old,x);
		vadd_p(Ad,r_old,r);
		dot_p(r,r, &r_dot);
		compt(r_dot,r_old_dot,beta,1);
		const_Vect_mult(beta, d_old,mult_d);
		vadd_p(mult_d,r,x);
		Copy(r,r_old);
		Copy(d,d_old);
	}

}

